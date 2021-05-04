#include "Calibration/HcalAlCaRecoProducers/interface/AlCaEcalHcalReadoutsProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

AlCaEcalHcalReadoutsProducer::AlCaEcalHcalReadoutsProducer(const edm::ParameterSet& iConfig) {
  tok_ho_ = consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInput"));
  tok_hf_ = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInput"));
  tok_hbhe_ = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInput"));

  //register your products
  put_hbhe_ = produces<HBHERecHitCollection>("HBHERecHitCollection");
  put_ho_ = produces<HORecHitCollection>("HORecHitCollection");
  put_hf_ = produces<HFRecHitCollection>("HFRecHitCollection");
}

// ------------ method called to produce the data  ------------
void AlCaEcalHcalReadoutsProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
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

  //Put selected information in the event
  iEvent.emplace(put_hbhe_, *hbhe);
  iEvent.emplace(put_ho_, *ho);
  iEvent.emplace(put_hf_, *hf);
}
