#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"

class GeneralTracksImporter : public BlockElementImporterBase {
public:
  GeneralTracksImporter(const edm::ParameterSet& conf,
		    edm::ConsumesCollector& sumes) :
    BlockElementImporterBase(conf,sumes),
    _src(sumes.consumes<reco::PFRecTrackCollection>(conf.getParameter<edm::InputTag>("source"))),
    _muons(sumes.consumes<reco::MuonCollection>(conf.getParameter<edm::InputTag>("muonSrc"))),
    _DPtovPtCut(conf.getParameter<std::vector<double> >("DPtOverPtCuts_byTrackAlgo")),
    _NHitCut(conf.getParameter<std::vector<unsigned> >("NHitCuts_byTrackAlgo")),
    _useIterTracking(conf.getParameter<bool>("useIterativeTracking")),
    _cleanBadConvBrems(conf.existsAs<bool>("cleanBadConvertedBrems") ? conf.getParameter<bool>("cleanBadConvertedBrems") : false),
    _debug(conf.getUntrackedParameter<bool>("debug",false)) {
    
    pfmu_ = std::unique_ptr<PFMuonAlgo>(new PFMuonAlgo());
    pfmu_->setParameters(conf);
    
  }
  
  void importToBlock( const edm::Event& ,
		      ElementList& ) const override;

private:
  bool goodPtResolution( const reco::TrackRef& trackref) const;
  int muAssocToTrack( const reco::TrackRef& trackref,
		      const edm::Handle<reco::MuonCollection>& muonh) const;

  edm::EDGetTokenT<reco::PFRecTrackCollection> _src;
  edm::EDGetTokenT<reco::MuonCollection> _muons;
  const std::vector<double> _DPtovPtCut;
  const std::vector<unsigned> _NHitCut;
  const bool _useIterTracking,_cleanBadConvBrems,_debug;

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
  e.getByToken(_src,tracks);
   edm::Handle<reco::MuonCollection> muons;
  e.getByToken(_muons,muons);
  elems.reserve(elems.size() + tracks->size());
  std::vector<bool> mask(tracks->size(),true);
  reco::MuonRef muonref;
  // remove converted brems with bad pT resolution if requested
  // this reproduces the old behavior of PFBlockAlgo
  if( _cleanBadConvBrems ) {
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
	    cRef.size() == 0 && dvRef.isNull() && v0Ref.isNull() ) {
	  // if the Pt resolution is bad we kill this element
	  if( !goodPtResolution( trkel->trackRef() ) ) {
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
  reco::PFBlockElementTrack* trkElem = NULL;
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
    if(thisIsAPotentialMuon || goodPtResolution( pftrackref->trackRef() ) ) {
      trkElem = new reco::PFBlockElementTrack( pftrackref );
      if (thisIsAPotentialMuon && _debug) {
	std::cout << "Potential Muon P " <<  pftrackref->trackRef()->p() 
		  << " pt " << pftrackref->trackRef()->p() << std::endl; 
      }
      if( muId != -1 ) trkElem->setMuonRef(muonref);
      elems.emplace_back(trkElem);
    }
  }
  elems.shrink_to_fit();
}

bool GeneralTracksImporter::
goodPtResolution( const reco::TrackRef& trackref) const {

  const double P = trackref->p();
  const double Pt = trackref->pt();
  const double DPt = trackref->ptError();
  const unsigned int NHit = 
    trackref->hitPattern().trackerLayersWithMeasurement();
  const unsigned int NLostHit = 
    trackref->hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
  const unsigned int LostHits = trackref->numberOfLostHits();
  const double sigmaHad = sqrt(1.20*1.20/P+0.06*0.06) / (1.+LostHits);

  // iteration 1,2,3,4,5 correspond to algo = 1/4,5,6,7,8,9
  unsigned int Algo = 0; 
  switch (trackref->algo()) {
  case reco::TrackBase::ctf:
  case reco::TrackBase::initialStep:
  case reco::TrackBase::lowPtTripletStep:
  case reco::TrackBase::pixelPairStep:
  case reco::TrackBase::jetCoreRegionalStep:
    Algo = 0;
    break;
  case reco::TrackBase::detachedTripletStep:
    Algo = 1;
    break;
  case reco::TrackBase::mixedTripletStep:
    Algo = 2;
    break;
  case reco::TrackBase::pixelLessStep:
    Algo = 3;
    break;
  case reco::TrackBase::tobTecStep:
    Algo = 4;
    break;
  case reco::TrackBase::muonSeededStepInOut:
  case reco::TrackBase::muonSeededStepOutIn:
    Algo = 5;
    break;
  case reco::TrackBase::hltIter0:
  case reco::TrackBase::hltIter1:
  case reco::TrackBase::hltIter2:
    Algo = _useIterTracking ? 0 : 0;
    break;
  case reco::TrackBase::hltIter3:
    Algo = _useIterTracking ? 1 : 0;
    break;
  case reco::TrackBase::hltIter4:
    Algo = _useIterTracking ? 2 : 0;
    break;
  case reco::TrackBase::hltIterX:
    Algo = _useIterTracking ? 0 : 0;
    break;
  default:
    Algo = _useIterTracking ? 6 : 0;
    break;
  }

  // Protection against 0 momentum tracks
  if ( P < 0.05 ) return false;

 
  if (_debug) std::cout << " PFBlockAlgo: PFrecTrack->Track Pt= "
		   << Pt << " DPt = " << DPt << std::endl;
  if ( ( _DPtovPtCut[Algo] > 0. && 
	 DPt/Pt > _DPtovPtCut[Algo]*sigmaHad ) || 
       NHit < _NHitCut[Algo] ) { 
    if (_debug) std::cout << " PFBlockAlgo: skip badly measured track"
		     << ", P = " << P 
		     << ", Pt = " << Pt 
		     << " DPt = " << DPt 
		     << ", N(hits) = " << NHit << " (Lost : " << LostHits << "/" << NLostHit << ")"
		     << ", Algo = " << Algo
		     << std::endl;
    if (_debug) std::cout << " cut is DPt/Pt < " << _DPtovPtCut[Algo] * sigmaHad << std::endl;
    if (_debug) std::cout << " cut is NHit >= " << _NHitCut[Algo] << std::endl;
    return false;
  }

  return true;
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
