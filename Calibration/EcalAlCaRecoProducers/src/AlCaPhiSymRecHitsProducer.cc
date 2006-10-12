#include "Calibration/EcalAlCaRecoProducers/interface/AlCaPhiSymRecHitsProducer.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

AlCaPhiSymRecHitsProducer::AlCaPhiSymRecHitsProducer(const edm::ParameterSet& iConfig)
{
  ecalHitsProducer_ = iConfig.getParameter< std::string > ("ecalRecHitsProducer");
  barrelHits_ = iConfig.getParameter< std::string > ("barrelHitCollection");
  endcapHits_ = iConfig.getParameter< std::string > ("endcapHitCollection");
  phiSymBarrelHits_ = iConfig.getParameter< std::string > ("phiSymBarrelHitCollection");
  phiSymEndcapHits_ = iConfig.getParameter< std::string > ("phiSymEndcapHitCollection");
  eCut_barl_ = iConfig.getParameter< double > ("eCut_barrel");
  eCut_endc_ = iConfig.getParameter< double > ("eCut_endcap");

  //register your products
  produces< EBRecHitCollection >(phiSymBarrelHits_);
  produces< EERecHitCollection >(phiSymEndcapHits_);
}


AlCaPhiSymRecHitsProducer::~AlCaPhiSymRecHitsProducer()
{
 

}


// ------------ method called to produce the data  ------------
void
AlCaPhiSymRecHitsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  Handle<EBRecHitCollection> barrelRecHitsHandle;
  Handle<EERecHitCollection> endcapRecHitsHandle;

  try {
    iEvent.getByLabel(ecalHitsProducer_,barrelHits_,barrelRecHitsHandle);
    iEvent.getByLabel(ecalHitsProducer_,endcapHits_,endcapRecHitsHandle);
  } catch ( std::exception& ex ) {
    LogDebug("") << "AlCaPhiSymRecHitsProducer: Error! can't get product!" << std::endl;
  }

  //Create empty output collections
  std::auto_ptr< EBRecHitCollection > phiSymEBRecHitCollection( new EBRecHitCollection );
  std::auto_ptr< EERecHitCollection > phiSymEERecHitCollection( new EERecHitCollection );

  //Select interesting EcalRecHits (barrel)
  EBRecHitCollection::const_iterator itb;
  for (itb=barrelRecHitsHandle->begin(); itb!=barrelRecHitsHandle->end(); itb++) {
    if (itb->energy() >= eCut_barl_) {
      phiSymEBRecHitCollection->push_back(*itb);
    }
  }

  //Select interesting EcalRecHits (endcaps)
  EERecHitCollection::const_iterator ite;
  for (ite=endcapRecHitsHandle->begin(); ite!=endcapRecHitsHandle->end(); ite++) {
    if (ite->energy() >= eCut_endc_) {
      phiSymEERecHitCollection->push_back(*ite);
    }
  }

  //Put selected information in the event
  iEvent.put( phiSymEBRecHitCollection, phiSymBarrelHits_);
  iEvent.put( phiSymEERecHitCollection, phiSymEndcapHits_);
  
}
