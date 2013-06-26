#include "RecoLocalCalo/EcalRecProducers/plugins/ESRecHitProducer.h"

#include "RecoLocalCalo/EcalRecProducers/interface/ESRecHitWorkerFactory.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ESRecHitProducer::ESRecHitProducer(edm::ParameterSet const& ps) {

  digiCollection_ = ps.getParameter<edm::InputTag>("ESdigiCollection");
  rechitCollection_ = ps.getParameter<std::string>("ESrechitCollection");
  produces<ESRecHitCollection>(rechitCollection_);
  
  std::string componentType = ps.getParameter<std::string>("algo");
  worker_ = ESRecHitWorkerFactory::get()->create( componentType, ps );
}

ESRecHitProducer::~ESRecHitProducer() {

  delete worker_;
}

void ESRecHitProducer::produce(edm::Event& e, const edm::EventSetup& es) {

  edm::Handle<ESDigiCollection> digiHandle;  
  const ESDigiCollection* digi=0;
  e.getByLabel( digiCollection_, digiHandle);
  if ( digiHandle.isValid() ) {
    digi = digiHandle.product();
    LogDebug("ESRecHitInfo") << "total # ESdigis: " << digi->size();
  } else {
    edm::LogError("ESRecHitInfo") << "Error! can't get the product " << digiCollection_;
  }
  
  // Create empty output
  std::auto_ptr<ESRecHitCollection> rec(new ESRecHitCollection );
  
  if ( digi ) {
    rec->reserve(digi->size()); 
    
    worker_->set( es );
    
    // run the algorithm
    for (ESDigiCollection::const_iterator i (digi->begin());
	 i!=digi->end(); i++) {    
      worker_->run( i, *rec );
    }
  }
  
  e.put(rec,rechitCollection_);
}

#include "FWCore/Framework/interface/MakerMacros.h"  
DEFINE_FWK_MODULE( ESRecHitProducer );

