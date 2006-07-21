#include "RecoParticleFlow/PFProducer/interface/PFProducer.h"

// #include "DataFormats/PFReco/interface/PFRecHit.h"
#include "DataFormats/PFReco/interface/PFLayer.h"
#include "DataFormats/PFReco/interface/PFCluster.h"
#include "DataFormats/PFReco/interface/PFClusterFwd.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace edm;

PFProducer::PFProducer(const edm::ParameterSet& iConfig)
{
  //register your products
  produces<reco::PFClusterCollection>();

  // dummy... just to be able to run
  // produces<reco::PFRecHitCollection >();
  
}


PFProducer::~PFProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.) 
}



void PFProducer::produce(edm::Event& iEvent, 
				const edm::EventSetup& iSetup) {

  cout<<"PFProducer::produce event: "<<iEvent.id().event()
      <<" in run "<<iEvent.id().run()<<endl;
  
  // std::vector<edm::Handle<HBHERecHitCollection> > hcalHandles;  

  edm::Handle<reco::TrackCollection> handleToTracks;
  try {
    // iEvent.getByType(handleToTracks);
    iEvent.getByLabel("CTFWMaterial", "", handleToTracks);
    cerr<<"got HCAL rechit handles"<<endl;
    
    const vector<reco::Track>& tracks = *handleToTracks;
    for(unsigned i=0; i<tracks.size(); i++) {
      cout<<"track "<<tracks[i].parameters().pt()<<" "<<endl;
    } 
  }
  catch (...) { 
    assert(0);
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(PFProducer)
