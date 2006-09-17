#include "PluginManager/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FastSimulation/CaloRecHitsProducer/interface/CaloRecHitsProducer.h"
#include "FastSimulation/CaloRecHitsProducer/interface/HcalRecHitsMaker.h"
#include "FastSimulation/CaloRecHitsProducer/interface/EcalBarrelRecHitsMaker.h"
#include "FastSimulation/CaloRecHitsProducer/interface/EcalEndcapRecHitsMaker.h"

#include <iostream>

CaloRecHitsProducer::CaloRecHitsProducer(edm::ParameterSet const & p):HcalRecHitsMaker_(NULL)
{    
  edm::ParameterSet RecHitsParameters = p.getParameter<edm::ParameterSet>("RecHitsFactory");
  EBrechitCollection_ = RecHitsParameters.getParameter<std::string>("EBrechitCollection");
  EErechitCollection_ = RecHitsParameters.getParameter<std::string>("EErechitCollection");
  
  produces<HBHERecHitCollection>();
  produces<HORecHitCollection>();
  produces<HFRecHitCollection>();
  produces<EBRecHitCollection>(EBrechitCollection_);
  produces<EERecHitCollection>(EErechitCollection_);
  

  HcalRecHitsMaker_ = new HcalRecHitsMaker(RecHitsParameters);
  EcalBarrelRecHitsMaker_ = new EcalBarrelRecHitsMaker(RecHitsParameters);
    //    EcalEndcapRecHitsMaker_ = new EcalEndcapRecHitsMaker(RecHitsParameters);
}

CaloRecHitsProducer::~CaloRecHitsProducer() 
{ 
  if (EcalBarrelRecHitsMaker_) delete EcalBarrelRecHitsMaker_;
  //  if (EcalEndcapRecHitsMaker_) delete EcalEndcapRecHitsMaker_;
  if (HcalRecHitsMaker_) delete HcalRecHitsMaker_; 
}

void CaloRecHitsProducer::beginJob(const edm::EventSetup & es)
{
  std::cout << " (Fast)RecHitsProducer initializing " << std::endl;
  HcalRecHitsMaker_->init(es);
}

void CaloRecHitsProducer::endJob()
{ 
    std::cout << " (Fast)RecHitsProducer terminating " << std::endl; 
}

void CaloRecHitsProducer::produce(edm::Event & iEvent, const edm::EventSetup & es)
{
  // create empty outputs for HCAL 
  // see RecoLocalCalo/HcalRecProducers/src/HcalSimpleReconstructor.cc
  
  std::auto_ptr<EBRecHitCollection> receb(new EBRecHitCollection);  // ECAL Barrel
  std::auto_ptr<EERecHitCollection> recee(new EERecHitCollection);  // ECAL Endcap

  std::auto_ptr<HBHERecHitCollection> rec1(new HBHERecHitCollection); // Barrel+Endcap
  std::auto_ptr<HORecHitCollection> rec2(new HORecHitCollection);     // Outer
  std::auto_ptr<HFRecHitCollection> rec3(new HFRecHitCollection);     // Forward

  EcalBarrelRecHitsMaker_->loadEcalBarrelRecHits(iEvent,*receb);
  EcalEndcapRecHitsMaker_->loadEcalEndcapRecHits(iEvent,*recee);

  HcalRecHitsMaker_->loadHcalRecHits(iEvent,*rec1,*rec2,*rec3);

  iEvent.put(receb,EBrechitCollection_);
  iEvent.put(recee,EErechitCollection_);
  iEvent.put(rec1);
  iEvent.put(rec2);
  iEvent.put(rec3);
}

DEFINE_FWK_MODULE(CaloRecHitsProducer)
