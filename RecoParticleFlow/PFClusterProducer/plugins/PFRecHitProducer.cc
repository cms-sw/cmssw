#include "RecoParticleFlow/PFClusterProducer/plugins/PFRecHitProducer.h"

#include <memory>

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"



using namespace std;
using namespace edm;


PFRecHitProducer::PFRecHitProducer(const edm::ParameterSet& iConfig)
{

    
  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  thresh_Barrel_ = 
    iConfig.getParameter<double>("thresh_Barrel");
  thresh_Endcap_ = 
    iConfig.getParameter<double>("thresh_Endcap");
    
    
  
  //register products
  produces<reco::PFRecHitCollection>();
  
}


void PFRecHitProducer::produce(edm::Event& iEvent, 
			       const edm::EventSetup& iSetup) {


  auto_ptr< vector<reco::PFRecHit> > recHits( new vector<reco::PFRecHit> ); 
  
  // fill the collection of rechits (see child classes)
  createRecHits( *recHits, iEvent, iSetup);

  iEvent.put( recHits );
}


PFRecHitProducer::~PFRecHitProducer() {}



//define this as a plug-in
// DEFINE_FWK_MODULE(PFRecHitProducer);

