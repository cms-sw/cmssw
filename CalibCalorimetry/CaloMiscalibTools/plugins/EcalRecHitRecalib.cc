
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/EcalRecHitRecalib.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalRecHitRecalib::EcalRecHitRecalib(const edm::ParameterSet& iConfig) {
  ecalHitsProducer_ = iConfig.getParameter<std::string>("ecalRecHitsProducer");
  barrelHits_ = iConfig.getParameter<std::string>("barrelHitCollection");
  endcapHits_ = iConfig.getParameter<std::string>("endcapHitCollection");
  RecalibBarrelHits_ = iConfig.getParameter<std::string>("RecalibBarrelHitCollection");
  RecalibEndcapHits_ = iConfig.getParameter<std::string>("RecalibEndcapHitCollection");
  refactor_ = iConfig.getUntrackedParameter<double>("Refactor", (double)1);
  refactor_mean_ = iConfig.getUntrackedParameter<double>("Refactor_mean", (double)1);

  //register your products
  produces<EBRecHitCollection>(RecalibBarrelHits_);
  produces<EERecHitCollection>(RecalibEndcapHits_);
}

EcalRecHitRecalib::~EcalRecHitRecalib() {}

// ------------ method called to produce the data  ------------
void EcalRecHitRecalib::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  Handle<EBRecHitCollection> barrelRecHitsHandle;
  Handle<EERecHitCollection> endcapRecHitsHandle;

  const EBRecHitCollection* EBRecHits = nullptr;
  const EERecHitCollection* EERecHits = nullptr;

  iEvent.getByLabel(ecalHitsProducer_, barrelHits_, barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    LogDebug("") << "EcalREcHitMiscalib: Error! can't get product!" << std::endl;
  } else {
    EBRecHits = barrelRecHitsHandle.product();  // get a ptr to the product
  }

  iEvent.getByLabel(ecalHitsProducer_, endcapHits_, endcapRecHitsHandle);
  if (!endcapRecHitsHandle.isValid()) {
    LogDebug("") << "EcalREcHitMiscalib: Error! can't get product!" << std::endl;
  } else {
    EERecHits = endcapRecHitsHandle.product();  // get a ptr to the product
  }

  //Create empty output collections
  auto RecalibEBRecHitCollection = std::make_unique<EBRecHitCollection>();
  auto RecalibEERecHitCollection = std::make_unique<EERecHitCollection>();

  // Intercalib constants
  edm::ESHandle<EcalIntercalibConstants> pIcal;
  iSetup.get<EcalIntercalibConstantsRcd>().get(pIcal);
  const EcalIntercalibConstants* ical = pIcal.product();

  if (EBRecHits) {
    //loop on all EcalRecHits (barrel)
    EBRecHitCollection::const_iterator itb;
    for (itb = EBRecHits->begin(); itb != EBRecHits->end(); ++itb) {
      // find intercalib constant for this xtal
      EcalIntercalibConstantMap::const_iterator icalit = ical->getMap().find(itb->id().rawId());
      EcalIntercalibConstant icalconst = -1;

      if (icalit != ical->getMap().end()) {
        icalconst = (*icalit);
        // edm::LogDebug("EcalRecHitRecalib") << "Found intercalib for xtal " << EBDetId(itb->id()) << " " << icalconst ;

      } else {
        edm::LogError("EcalRecHitRecalib") << "No intercalib const found for xtal " << EBDetId(itb->id())
                                           << "! something wrong with EcalIntercalibConstants in your DB? ";
      }

      // make the rechit with rescaled energy and put in the output collection
      icalconst = refactor_mean_ +
                  (icalconst - refactor_mean_) * refactor_;  //apply additional scaling factor (works if gaussian)
      EcalRecHit aHit(itb->id(), itb->energy() * icalconst, itb->time());

      RecalibEBRecHitCollection->push_back(aHit);
    }
  }

  if (EERecHits) {
    //loop on all EcalRecHits (barrel)
    EERecHitCollection::const_iterator ite;
    for (ite = EERecHits->begin(); ite != EERecHits->end(); ++ite) {
      // find intercalib constant for this xtal
      EcalIntercalibConstantMap::const_iterator icalit = ical->getMap().find(ite->id().rawId());
      EcalIntercalibConstant icalconst = -1;

      if (icalit != ical->getMap().end()) {
        icalconst = (*icalit);
        // edm:: LogDebug("EcalRecHitRecalib") << "Found intercalib for xtal " << EEDetId(ite->id()) << " " << icalconst ;
      } else {
        edm::LogError("EcalRecHitRecalib") << "No intercalib const found for xtal " << EEDetId(ite->id())
                                           << "! something wrong with EcalIntercalibConstants in your DB? ";
      }

      // make the rechit with rescaled energy and put in the output collection

      icalconst = refactor_mean_ +
                  (icalconst - refactor_mean_) * refactor_;  //apply additional scaling factor (works if gaussian)
      EcalRecHit aHit(ite->id(), ite->energy() * icalconst, ite->time());

      RecalibEERecHitCollection->push_back(aHit);
    }
  }

  //Put Recalibrated rechit in the event
  iEvent.put(std::move(RecalibEBRecHitCollection), RecalibBarrelHits_);
  iEvent.put(std::move(RecalibEERecHitCollection), RecalibEndcapHits_);
}
