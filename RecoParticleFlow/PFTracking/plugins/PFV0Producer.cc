#include <memory>
#include "RecoParticleFlow/PFTracking/interface/PFV0Producer.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0Fwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
using namespace std;
using namespace edm;
using namespace reco;
PFV0Producer::PFV0Producer(const ParameterSet& iConfig):
  conf_(iConfig)
{

  produces<reco::PFV0Collection>();
}

PFV0Producer::~PFV0Producer()
{
}

void
PFV0Producer::produce(Event& iEvent, const EventSetup& iSetup)
{
  LogDebug("PFV0Producer")<<"START event: "<<iEvent.id().event()
			  <<" in run "<<iEvent.id().run();
  //create the empty collections 
  auto_ptr< PFV0Collection > 
    pfV0Coll (new PFV0Collection);
  
  Handle< PFRecTrackCollection > recTracks;
  iEvent.getByLabel(conf_.getParameter<InputTag>("PFTrackColl"), recTracks);
  
  
  vector<InputTag> V0list=conf_.getParameter< vector < InputTag > >("V0List");
  for (uint il=0; il<V0list.size(); il++){
    Handle<VertexCompositeCandidateCollection> V0coll;
    iEvent.getByLabel(V0list[il],V0coll);
     LogDebug("PFV0Producer")<<V0list[il]<<" contains "<<V0coll->size()<<" V0 candidates ";
    for (uint iv=0;iv<V0coll->size();iv++){
      VertexCompositeCandidateRef V0(V0coll, iv);
      vector<TrackRef> Tracks;
      vector<PFRecTrackRef> PFTracks;
      for( uint ndx = 0; ndx < V0->numberOfDaughters(); ndx++ ) {
	
	Tracks.push_back( (dynamic_cast<const RecoChargedCandidate*>(V0->daughter(ndx)))->track() );
	for (uint ipt=0;ipt<recTracks->size();ipt++){
	  PFRecTrackRef pft(recTracks,ipt);
	  if((dynamic_cast<const RecoChargedCandidate*>(V0->daughter(ndx)))->track()==pft->trackRef())
	    PFTracks.push_back(pft);
	  
	}
      }
      if ((PFTracks.size()==2)&&(Tracks.size()==2)){
	pfV0Coll->push_back(PFV0(V0,PFTracks,Tracks));
      }
	
      

    }
  }
  
  
  iEvent.put(pfV0Coll);
}



// ------------ method called once each job just after ending the event loop  ------------
void 
PFV0Producer::endJob() {
}
