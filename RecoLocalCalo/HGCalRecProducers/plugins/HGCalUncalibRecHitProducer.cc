#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalUncalibRecHitProducer.h"

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalUncalibRecHitWorkerFactory.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

HGCalUncalibRecHitProducer::HGCalUncalibRecHitProducer(const edm::ParameterSet& ps)
    : eeDigiCollection_(consumes<HGCalDigiCollection>(ps.getParameter<edm::InputTag>("HGCEEdigiCollection"))),
      hefDigiCollection_(consumes<HGCalDigiCollection>(ps.getParameter<edm::InputTag>("HGCHEFdigiCollection"))),
      hebDigiCollection_(consumes<HGCalDigiCollection>(ps.getParameter<edm::InputTag>("HGCHEBdigiCollection"))),
      hfnoseDigiCollection_(consumes<HGCalDigiCollection>(ps.getParameter<edm::InputTag>("HGCHFNosedigiCollection"))),
      eeHitCollection_(ps.getParameter<std::string>("HGCEEhitCollection")),
      hefHitCollection_(ps.getParameter<std::string>("HGCHEFhitCollection")),
      hebHitCollection_(ps.getParameter<std::string>("HGCHEBhitCollection")),
      hfnoseHitCollection_(ps.getParameter<std::string>("HGCHFNosehitCollection")),
      worker_{HGCalUncalibRecHitWorkerFactory::get()->create(
          ps.getParameter<std::string>("algo"), ps, consumesCollector())} {
  produces<HGCeeUncalibratedRecHitCollection>(eeHitCollection_);
  produces<HGChefUncalibratedRecHitCollection>(hefHitCollection_);
  produces<HGChebUncalibratedRecHitCollection>(hebHitCollection_);
  produces<HGChfnoseUncalibratedRecHitCollection>(hfnoseHitCollection_);
}

HGCalUncalibRecHitProducer::~HGCalUncalibRecHitProducer() {}

void HGCalUncalibRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  using namespace edm;

  // tranparently get things from event setup
  worker_->set(es);

  // prepare output
  auto eeUncalibRechits = std::make_unique<HGCeeUncalibratedRecHitCollection>();
  auto hefUncalibRechits = std::make_unique<HGChefUncalibratedRecHitCollection>();
  auto hebUncalibRechits = std::make_unique<HGChebUncalibratedRecHitCollection>();
  auto hfnoseUncalibRechits = std::make_unique<HGChfnoseUncalibratedRecHitCollection>();

  // loop over HGCEE digis
  edm::Handle<HGCalDigiCollection> pHGCEEDigis;
  evt.getByToken(eeDigiCollection_, pHGCEEDigis);
  const HGCalDigiCollection* eeDigis = pHGCEEDigis.product();
  eeUncalibRechits->reserve(eeDigis->size());
  for (auto itdg = eeDigis->begin(); itdg != eeDigis->end(); ++itdg) {
    worker_->runHGCEE(itdg, *eeUncalibRechits);
  }

  // loop over HGCHEsil digis
  edm::Handle<HGCalDigiCollection> pHGCHEFDigis;
  evt.getByToken(hefDigiCollection_, pHGCHEFDigis);
  const HGCalDigiCollection* hefDigis = pHGCHEFDigis.product();
  hefUncalibRechits->reserve(hefDigis->size());
  for (auto itdg = hefDigis->begin(); itdg != hefDigis->end(); ++itdg) {
    worker_->runHGCHEsil(itdg, *hefUncalibRechits);
  }

  // loop over HGCHEscint digis
  edm::Handle<HGCalDigiCollection> pHGCHEBDigis;
  evt.getByToken(hebDigiCollection_, pHGCHEBDigis);
  const HGCalDigiCollection* hebDigis = pHGCHEBDigis.product();
  hebUncalibRechits->reserve(hebDigis->size());
  for (auto itdg = hebDigis->begin(); itdg != hebDigis->end(); ++itdg) {
    worker_->runHGCHEscint(itdg, *hebUncalibRechits);
  }

  // loop over HFNose digis
  edm::Handle<HGCalDigiCollection> pHGCHFNoseDigis;
  evt.getByToken(hfnoseDigiCollection_, pHGCHFNoseDigis);
  if (pHGCHFNoseDigis.isValid()) {
    const HGCalDigiCollection* hfnoseDigis = pHGCHFNoseDigis.product();
    if (!(hfnoseDigis->empty())) {
      hfnoseUncalibRechits->reserve(hfnoseDigis->size());
      for (auto itdg = hfnoseDigis->begin(); itdg != hfnoseDigis->end(); ++itdg)
        worker_->runHGCHFNose(itdg, *hfnoseUncalibRechits);
    }
  }

  // put the collection of recunstructed hits in the event
  evt.put(std::move(eeUncalibRechits), eeHitCollection_);
  evt.put(std::move(hefUncalibRechits), hefHitCollection_);
  evt.put(std::move(hebUncalibRechits), hebHitCollection_);
  if (pHGCHFNoseDigis.isValid())
    evt.put(std::move(hfnoseUncalibRechits), hfnoseHitCollection_);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HGCalUncalibRecHitProducer);
