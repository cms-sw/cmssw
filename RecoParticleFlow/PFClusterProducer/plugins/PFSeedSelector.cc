#include "RecoParticleFlow/PFClusterProducer/plugins/PFSeedSelector.h"
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace edm;


PFSeedSelector::PFSeedSelector(const edm::ParameterSet& iConfig):
  hits_(consumes<reco::PFRecHitCollection>(iConfig.getParameter<edm::InputTag>("src")))
{
  produces<reco::PFRecHitCollection>();
}


void PFSeedSelector::produce(edm::Event& iEvent, 
			       const edm::EventSetup& iSetup) {

  edm::Handle<reco::PFRecHitCollection> hits; 
  iEvent.getByToken(hits_,hits);
  std::auto_ptr<reco::PFRecHitCollection> out(new reco::PFRecHitCollection);

  for (const auto& hit : *hits) {
    bool maximum=true;
    for (const auto& neighbour : hit.neighbours4()) {
      if (hit.energy()<neighbour->energy()) {
	maximum=false;
	break;
      }
    }
      if (maximum)
	out->push_back(hit);
  }



  iEvent.put( out);

}

PFSeedSelector::~PFSeedSelector() {}

// ------------ method called once each job just before starting event loop  ------------
void 
PFSeedSelector::beginRun(const edm::Run& run,
			   const EventSetup& es) {

 
}


