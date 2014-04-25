#include "RecoParticleFlow/PFClusterProducer/plugins/PFArborLinker.h"
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace edm;


PFArborLinker::PFArborLinker(const edm::ParameterSet& iConfig):
  hits_(consumes<reco::PFRecHitCollection>(iConfig.getParameter<edm::InputTag>("src")))
{
  produces<reco::PFClusterCollection>("allLinks");
}


void PFArborLinker::produce(edm::Event& iEvent, 
			    const edm::EventSetup& iSetup) {

  //Read seeds
  edm::Handle<reco::PFRecHitCollection> hits; 
  iEvent.getByToken(hits_,hits);
  std::auto_ptr<reco::PFClusterCollection> out(new reco::PFClusterCollection);
  
  for (unsigned int i=0;i<hits->size();++i) {
    reco::PFCluster c(PFLayer::HCAL_BARREL1,hits->at(i).energy(),hits->at(i).position().x(),
		     hits->at(i).position().y(),hits->at(i).position().z());

    reco::PFRecHitRef hitRef(hits,i);
    reco::PFRecHitFraction fraction(hitRef,1.0);
    c.addRecHitFraction(fraction);
    
    out->push_back(c);
      

  }


  iEvent.put( out,"allLinks");

}

PFArborLinker::~PFArborLinker() {}

// ------------ method called once each job just before starting event loop  ------------
void 
PFArborLinker::beginRun(const edm::Run& run,
			   const EventSetup& es) {

 
}


