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

// this doesn't actually import anything, 
// but rather applies time stamps to tracks after they are all inserted

class TrackTimingImporter : public BlockElementImporterBase {
public:
  TrackTimingImporter(const edm::ParameterSet& conf,
		    edm::ConsumesCollector& sumes) :
    BlockElementImporterBase(conf,sumes),    
    srcTime_( sumes.consumes<edm::ValueMap<float> >(conf.getParameter<edm::InputTag>("timeValueMap")) ),
    srcTimeError_( sumes.consumes<edm::ValueMap<float> >(conf.getParameter<edm::InputTag>("timeErrorMap")) ),
    debug_(conf.getUntrackedParameter<bool>("debug",false)) {    
  }
  
  void importToBlock( const edm::Event& ,
		      ElementList& ) const override;

private:
    
  edm::EDGetTokenT<edm::ValueMap<float> > srcTime_, srcTimeError_;
  const bool debug_;
  
};

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, 
		  TrackTimingImporter, 
		  "TrackTimingImporter");

void TrackTimingImporter::
importToBlock( const edm::Event& e, 
	       BlockElementImporterBase::ElementList& elems ) const {
  typedef BlockElementImporterBase::ElementList::value_type ElementType;  
  
  edm::Handle<edm::ValueMap<float> > timeH, timeErrH;
  
  e.getByToken(srcTime_, timeH);
  e.getByToken(srcTimeError_, timeErrH);
    
  auto TKs_end = std::partition(elems.begin(),elems.end(),
				[](const ElementType& a){
			        return a->type() == reco::PFBlockElement::TRACK;
				});  
  auto btk_elems = elems.begin();  
  // now we actually insert tracks, again tagging muons along the way
  reco::PFRecTrackRef pftrackref;  
  for( auto track = btk_elems;  track != TKs_end; ++track) {
    const auto& ref = (*track)->trackRef();
    (*track)->setTime( (*timeH)[ref], (*timeErrH)[ref] );
    if( debug_ ) {
      edm::LogInfo("TrackTimingImporter") 
      	<< "Track with pT / eta " << ref->pt() << " / " << ref->eta() 
	<< " has time: " << (*track)->time() << " +/- " << (*track)->timeError() << std::endl;
    }
    
  }
}
