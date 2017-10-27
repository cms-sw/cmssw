#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackAlgoTools.h"

class GeneralTracksImporter : public BlockElementImporterBase {
public:
  GeneralTracksImporter(const edm::ParameterSet& conf,
		    edm::ConsumesCollector& sumes) :
    BlockElementImporterBase(conf,sumes),
    src_(sumes.consumes<reco::PFRecTrackCollection>(conf.getParameter<edm::InputTag>("source"))),
    muons_(sumes.consumes<reco::MuonCollection>(conf.getParameter<edm::InputTag>("muonSrc"))),
    DPtovPtCut_(conf.getParameter<std::vector<double> >("DPtOverPtCuts_byTrackAlgo")),
    NHitCut_(conf.getParameter<std::vector<unsigned> >("NHitCuts_byTrackAlgo")),    
    useIterTracking_(conf.getParameter<bool>("useIterativeTracking")),
    cleanBadConvBrems_(conf.existsAs<bool>("cleanBadConvertedBrems") ? conf.getParameter<bool>("cleanBadConvertedBrems") : false),
    debug_(conf.getUntrackedParameter<bool>("debug",false)) {
    
    pfmu_ = std::unique_ptr<PFMuonAlgo>(new PFMuonAlgo());
    pfmu_->setParameters(conf);
    
  }
  
  void importToBlock( const edm::Event& ,
		      ElementList& ) const override;

private:
  int muAssocToTrack( const reco::TrackRef& trackref,
		      const edm::Handle<reco::MuonCollection>& muonh) const;

  edm::EDGetTokenT<reco::PFRecTrackCollection> src_;
  edm::EDGetTokenT<reco::MuonCollection> muons_;
  const std::vector<double> DPtovPtCut_;
  const std::vector<unsigned> NHitCut_;
  const bool useIterTracking_,cleanBadConvBrems_,debug_;

  std::unique_ptr<PFMuonAlgo> pfmu_;

};

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, 
		  GeneralTracksImporter, 
		  "GeneralTracksImporter");

void GeneralTracksImporter::
importToBlock( const edm::Event& e, 
	       BlockElementImporterBase::ElementList& elems ) const {
  typedef BlockElementImporterBase::ElementList::value_type ElementType;  
  edm::Handle<reco::PFRecTrackCollection> tracks;
  e.getByToken(src_,tracks);
   edm::Handle<reco::MuonCollection> muons;
  e.getByToken(muons_,muons);
  elems.reserve(elems.size() + tracks->size());
  std::vector<bool> mask(tracks->size(),true);
  reco::MuonRef muonref;
  
  // remove converted brems with bad pT resolution if requested
  // this reproduces the old behavior of PFBlockAlgo
  if( cleanBadConvBrems_ ) {
    auto itr = elems.begin();
    while( itr != elems.end() ) {
      if( (*itr)->type() == reco::PFBlockElement::TRACK ) {
	const reco::PFBlockElementTrack* trkel =
	  static_cast<reco::PFBlockElementTrack*>(itr->get());
	const reco::ConversionRefVector& cRef = trkel->convRefs();
	const reco::PFDisplacedTrackerVertexRef& dvRef = 
	  trkel->displacedVertexRef(reco::PFBlockElement::T_FROM_DISP);
	const reco::VertexCompositeCandidateRef& v0Ref =
	  trkel->V0Ref();
	// if there is no displaced vertex reference  and it is marked
	// as a conversion it's gotta be a converted brem
	if( trkel->trackType(reco::PFBlockElement::T_FROM_GAMMACONV) &&
	    cRef.empty() && dvRef.isNull() && v0Ref.isNull() ) {
	  // if the Pt resolution is bad we kill this element
	  if( !PFTrackAlgoTools::goodPtResolution( trkel->trackRef(), DPtovPtCut_, NHitCut_, useIterTracking_, debug_ ) ) {
	    itr = elems.erase(itr);
	    continue;
	  }
	}
      }
      ++itr;
    } // loop on existing elements
  }
  // preprocess existing tracks in the element list and create a mask
  // so that we do not import tracks twice, tag muons we find 
  // in this collection  
  auto TKs_end = std::partition(elems.begin(),elems.end(),
				[](const ElementType& a){
			        return a->type() == reco::PFBlockElement::TRACK;
				});  
  auto btk_elems = elems.begin();
  auto btrack = tracks->cbegin();
  auto etrack = tracks->cend();
  for( auto track = btrack;  track != etrack; ++track) {
    auto tk_elem = std::find_if(btk_elems,TKs_end,
				[&](const ElementType& a){
				  return ( a->trackRef() == 
					   track->trackRef() );
				});
    if( tk_elem != TKs_end ) {
      mask[std::distance(tracks->cbegin(),track)] = false;
      // check and update if this track is a muon
      const int muId = muAssocToTrack( (*tk_elem)->trackRef(), muons );
      if( muId != -1 ) {
	muonref= reco::MuonRef( muons, muId );
	if( PFMuonAlgo::isLooseMuon(muonref) || PFMuonAlgo::isMuon(muonref) ) {
	  static_cast<reco::PFBlockElementTrack*>(tk_elem->get())->setMuonRef(muonref);
	}
      }
    }    
  }
  // now we actually insert tracks, again tagging muons along the way
  reco::PFRecTrackRef pftrackref;  
  reco::PFBlockElementTrack* trkElem = nullptr;
  for( auto track = btrack;  track != etrack; ++track) {
    const unsigned idx = std::distance(btrack,track);
    // since we already set muon refs in the previously imported tracks,
    // here we can skip everything that is already imported 
    if( !mask[idx] ) continue; 
    muonref = reco::MuonRef();
    pftrackref = reco::PFRecTrackRef(tracks,idx);    
    // Get the eventual muon associated to this track
    const int muId = muAssocToTrack( pftrackref->trackRef(), muons );
    bool thisIsAPotentialMuon = false;
    if( muId != -1 ) {
      muonref= reco::MuonRef( muons, muId );
      thisIsAPotentialMuon = ( (pfmu_->hasValidTrack(muonref,true)&&PFMuonAlgo::isLooseMuon(muonref)) || 
			       (pfmu_->hasValidTrack(muonref,false)&&PFMuonAlgo::isMuon(muonref)));
    }
    if(thisIsAPotentialMuon || PFTrackAlgoTools::goodPtResolution( pftrackref->trackRef(), DPtovPtCut_, NHitCut_, useIterTracking_, debug_ ) ) {
      trkElem = new reco::PFBlockElementTrack( pftrackref );
      if (thisIsAPotentialMuon && debug_) {
	std::cout << "Potential Muon P " <<  pftrackref->trackRef()->p() 
		  << " pt " << pftrackref->trackRef()->p() << std::endl; 
      }
      if( muId != -1 ) trkElem->setMuonRef(muonref);      
      elems.emplace_back(trkElem);
    }
  }
  elems.shrink_to_fit();
}

int GeneralTracksImporter::
muAssocToTrack( const reco::TrackRef& trackref,
		const edm::Handle<reco::MuonCollection>& muonh) const {
  auto muon = std::find_if(muonh->cbegin(),muonh->cend(),
			   [&](const reco::Muon& m) {
			     return ( m.track().isNonnull() && 
				      m.track() == trackref    );
			   });  
  return ( muon != muonh->cend() ? std::distance(muonh->cbegin(),muon) : -1 );
}
