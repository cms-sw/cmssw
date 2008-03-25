#include "RecoLocalCalo/EcalRecProducers/interface/ESRecHitProducer.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

// ESRecHitProducer author : Chia-Ming, Kuo

ESRecHitProducer::ESRecHitProducer(edm::ParameterSet const& ps) : theGeometry(0)
{
  digiCollection_ = ps.getParameter<edm::InputTag>("ESdigiCollection");
  rechitCollection_ = ps.getParameter<std::string>("ESrechitCollection");
  produces<ESRecHitCollection>(rechitCollection_);
  
  //These should be taken from a DB
  int ESGain = ps.getUntrackedParameter<int>("ESGain", 1);
  int ESBaseline = ps.getUntrackedParameter<int>("ESBaseline", 1000);
  double ESMIPADC = ps.getUntrackedParameter<double>("ESMIPADC", 9);
  double ESMIPkeV = ps.getUntrackedParameter<double>("ESMIPkeV", 81.08);

  algo_ = new ESRecHitSimAlgo(ESGain, ESBaseline, ESMIPADC, ESMIPkeV); 
}

ESRecHitProducer::~ESRecHitProducer() {
  delete algo_;
}

void ESRecHitProducer::produce(edm::Event& e, const edm::EventSetup& es)
{
  checkGeometry(es);

  // Get input
  edm::Handle<ESDigiCollection> digiHandle;  
  const ESDigiCollection* digi=0;
  //evt.getByLabel( digiProducer_, digiCollection_, pDigis);
  e.getByLabel( digiCollection_, digiHandle);
  digi=digiHandle.product();

  edm::LogInfo("ESRecHitInfo") << "total # ESdigis: " << digi->size() ;  
  // Create empty output
  std::auto_ptr<ESRecHitCollection> rec(new EcalRecHitCollection());

  // run the algorithm
  ESDigiCollection::const_iterator i;
  for (i=digi->begin(); i!=digi->end(); i++) {    
    rec->push_back(algo_->reconstruct(*i, false));
  }

  e.put(rec,rechitCollection_);
}

void ESRecHitProducer::checkGeometry(const edm::EventSetup & es) 
{
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> hGeometry;
  es.get<IdealGeometryRecord>().get(hGeometry);
  
  const CaloGeometry *pGeometry = &*hGeometry;
  
  // see if we need to update
  if(pGeometry != theGeometry) {
    theGeometry = pGeometry;
    updateGeometry();
  }
}

void ESRecHitProducer::updateGeometry() 
{
  algo_->setGeometry(theGeometry);
}



