#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/EcalRecHitRecalib.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalRecHitRecalib::EcalRecHitRecalib(const edm::ParameterSet& iConfig)
    : ecalHitsProducer_(iConfig.getParameter<std::string>("ecalRecHitsProducer")),
      barrelHits_(iConfig.getParameter<std::string>("barrelHitCollection")),
      endcapHits_(iConfig.getParameter<std::string>("endcapHitCollection")),
      recalibBarrelHits_(iConfig.getParameter<std::string>("RecalibBarrelHitCollection")),
      recalibEndcapHits_(iConfig.getParameter<std::string>("RecalibEndcapHitCollection")),
      refactor_(iConfig.getUntrackedParameter<double>("Refactor", (double)1)),
      refactor_mean_(iConfig.getUntrackedParameter<double>("Refactor_mean", (double)1)),
      ebRecHitToken_(consumes<EBRecHitCollection>(edm::InputTag(ecalHitsProducer_, barrelHits_))),
      eeRecHitToken_(consumes<EERecHitCollection>(edm::InputTag(ecalHitsProducer_, endcapHits_))),
      intercalibConstsToken_(esConsumes()),
      barrelHitsToken_(produces<EBRecHitCollection>(recalibBarrelHits_)),
      endcapHitsToken_(produces<EERecHitCollection>(recalibEndcapHits_)) {}

EcalRecHitRecalib::~EcalRecHitRecalib() {}

// ------------ method called to produce the data  ------------
void EcalRecHitRecalib::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  Handle<EBRecHitCollection> barrelRecHitsHandle;
  Handle<EERecHitCollection> endcapRecHitsHandle;

  const EBRecHitCollection* EBRecHits = nullptr;
  const EERecHitCollection* EERecHits = nullptr;

  iEvent.getByToken(ebRecHitToken_, barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    LogDebug("") << "EcalREcHitMiscalib: Error! can't get product!" << std::endl;
  } else {
    EBRecHits = barrelRecHitsHandle.product();  // get a ptr to the product
  }

  iEvent.getByToken(eeRecHitToken_, endcapRecHitsHandle);
  if (!endcapRecHitsHandle.isValid()) {
    LogDebug("") << "EcalREcHitMiscalib: Error! can't get product!" << std::endl;
  } else {
    EERecHits = endcapRecHitsHandle.product();  // get a ptr to the product
  }

  //Create empty output collections
  auto RecalibEBRecHitCollection = std::make_unique<EBRecHitCollection>();
  auto RecalibEERecHitCollection = std::make_unique<EERecHitCollection>();

  // Intercalib constants
  const EcalIntercalibConstants& ical = iSetup.getData(intercalibConstsToken_);

  if (EBRecHits) {
    //loop on all EcalRecHits (barrel)
    EBRecHitCollection::const_iterator itb;
    for (itb = EBRecHits->begin(); itb != EBRecHits->end(); ++itb) {
      // find intercalib constant for this xtal
      EcalIntercalibConstantMap::const_iterator icalit = ical.getMap().find(itb->id().rawId());
      EcalIntercalibConstant icalconst = -1;

      if (icalit != ical.getMap().end()) {
        icalconst = (*icalit);

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
      EcalIntercalibConstantMap::const_iterator icalit = ical.getMap().find(ite->id().rawId());
      EcalIntercalibConstant icalconst = -1;

      if (icalit != ical.getMap().end()) {
        icalconst = (*icalit);
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
  iEvent.put(barrelHitsToken_, std::move(RecalibEBRecHitCollection));
  iEvent.put(endcapHitsToken_, std::move(RecalibEERecHitCollection));
}
