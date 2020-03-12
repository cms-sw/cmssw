#include "Calibration/HcalAlCaRecoProducers/interface/AlCaEcalHcalReadoutsProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

AlCaEcalHcalReadoutsProducer::AlCaEcalHcalReadoutsProducer(const edm::ParameterSet& iConfig) {
  tok_ho_ = consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInput"));
  tok_hf_ = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInput"));
  tok_hbhe_ = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInput"));

  //register your products
  produces<HBHERecHitCollection>("HBHERecHitCollection");
  produces<HORecHitCollection>("HORecHitCollection");
  produces<HFRecHitCollection>("HFRecHitCollection");
}

AlCaEcalHcalReadoutsProducer::~AlCaEcalHcalReadoutsProducer() {}

// ------------ method called to produce the data  ------------
void AlCaEcalHcalReadoutsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  edm::Handle<HBHERecHitCollection> hbhe;
  edm::Handle<HORecHitCollection> ho;
  edm::Handle<HFRecHitCollection> hf;

  iEvent.getByToken(tok_hbhe_, hbhe);
  if (!hbhe.isValid()) {
    LogDebug("") << "AlCaEcalHcalReadoutProducer: Error! can't get hbhe product!" << std::endl;
    return;
  }

  iEvent.getByToken(tok_ho_, ho);
  if (!ho.isValid()) {
    LogDebug("") << "AlCaEcalHcalReadoutProducer: Error! can't get ho product!" << std::endl;
  }

  iEvent.getByToken(tok_hf_, hf);
  if (!hf.isValid()) {
    LogDebug("") << "AlCaEcalHcalReadoutProducer: Error! can't get hf product!" << std::endl;
  }
  //Create empty output collections

  auto miniHBHERecHitCollection = std::make_unique<HBHERecHitCollection>();
  auto miniHORecHitCollection = std::make_unique<HORecHitCollection>();
  auto miniHFRecHitCollection = std::make_unique<HFRecHitCollection>();

  const HBHERecHitCollection Hithbhe = *(hbhe.product());
  for (HBHERecHitCollection::const_iterator hbheItr = Hithbhe.begin(); hbheItr != Hithbhe.end(); hbheItr++) {
    miniHBHERecHitCollection->push_back(*hbheItr);
  }
  const HORecHitCollection Hitho = *(ho.product());
  for (HORecHitCollection::const_iterator hoItr = Hitho.begin(); hoItr != Hitho.end(); hoItr++) {
    miniHORecHitCollection->push_back(*hoItr);
  }

  const HFRecHitCollection Hithf = *(hf.product());
  for (HFRecHitCollection::const_iterator hfItr = Hithf.begin(); hfItr != Hithf.end(); hfItr++) {
    miniHFRecHitCollection->push_back(*hfItr);
  }

  //Put selected information in the event
  iEvent.put(std::move(miniHBHERecHitCollection), "HBHERecHitCollection");
  iEvent.put(std::move(miniHORecHitCollection), "HORecHitCollection");
  iEvent.put(std::move(miniHFRecHitCollection), "HFRecHitCollection");
}
