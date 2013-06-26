#include <memory>
#include "RecoParticleFlow/PFTracking/plugins/PFV0Producer.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0Fwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackTransformer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"

using namespace std;
using namespace edm;
using namespace reco;
PFV0Producer::PFV0Producer(const ParameterSet& iConfig):
  pfTransformer_(0)
{

  produces<reco::PFV0Collection>();
  produces<reco::PFRecTrackCollection>();

  V0list_ = 
    iConfig.getParameter< vector < InputTag > >("V0List");
}

PFV0Producer::~PFV0Producer()
{
  delete pfTransformer_;
}

void
PFV0Producer::produce(Event& iEvent, const EventSetup& iSetup)
{
  LogDebug("PFV0Producer")<<"START event: "<<iEvent.id().event()
			  <<" in run "<<iEvent.id().run();
  //create the empty collections 
  auto_ptr< PFV0Collection > pfV0Coll (new PFV0Collection);

  auto_ptr<reco::PFRecTrackCollection> pfV0RecTrackColl(new reco::PFRecTrackCollection);


  reco::PFRecTrackRefProd pfTrackRefProd = iEvent.getRefBeforePut<reco::PFRecTrackCollection>();
  int idx = 0;

  for (unsigned int il=0; il<V0list_.size(); il++){
    Handle<VertexCompositeCandidateCollection> V0coll;
    iEvent.getByLabel(V0list_[il],V0coll);
    LogDebug("PFV0Producer")<<V0list_[il]<<" contains "<<V0coll->size()<<" V0 candidates ";
    for (unsigned int iv=0;iv<V0coll->size();iv++){
      VertexCompositeCandidateRef V0(V0coll, iv);
      vector<TrackRef> Tracks;
      vector<PFRecTrackRef> PFTracks;
      for( unsigned int ndx = 0; ndx < V0->numberOfDaughters(); ndx++ ) {
	
	Tracks.push_back( (dynamic_cast<const RecoChargedCandidate*>(V0->daughter(ndx)))->track() );
	TrackRef trackRef = (dynamic_cast<const RecoChargedCandidate*>(V0->daughter(ndx)))->track();

	reco::PFRecTrack pfRecTrack( trackRef->charge(), 
				   reco::PFRecTrack::KF, 
				   trackRef.key(), 
				   trackRef );


	Trajectory FakeTraj;
	bool valid = pfTransformer_->addPoints( pfRecTrack, *trackRef, FakeTraj);
	if(valid) {
	  PFTracks.push_back(reco::PFRecTrackRef( pfTrackRefProd, idx++));
	  pfV0RecTrackColl->push_back(pfRecTrack);
	  
	}
      }
      if ((PFTracks.size()==2)&&(Tracks.size()==2)){
	pfV0Coll->push_back(PFV0(V0,PFTracks,Tracks));
      }
	
      

    }
  }
  
  
  iEvent.put(pfV0Coll);
  iEvent.put(pfV0RecTrackColl);
}

// ------------ method called once each job just before starting event loop  ------------
void 
PFV0Producer::beginRun(const edm::Run& run,
					   const EventSetup& iSetup)
{
  ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
  pfTransformer_= new PFTrackTransformer(math::XYZVector(magneticField->inTesla(GlobalPoint(0,0,0))));
  pfTransformer_->OnlyProp();
}


// ------------ method called once each job just after ending the event loop  ------------
void 
PFV0Producer::endRun(const edm::Run& run,
		     const EventSetup& iSetup) {
  delete pfTransformer_;
  pfTransformer_=nullptr;
}
