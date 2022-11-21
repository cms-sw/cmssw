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
      ee_geometry_token_(esConsumes(edm::ESInputTag("", "HGCalEESensitive"))),
      hef_geometry_token_(esConsumes(edm::ESInputTag("", "HGCalHESiliconSensitive"))),
      heb_geometry_token_(esConsumes(edm::ESInputTag("", "HGCalHEScintillatorSensitive"))),
      hfnose_geometry_token_(esConsumes(edm::ESInputTag("", "HGCalHFNoseSensitive"))),
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

  // prepare output
  auto eeUncalibRechits = std::make_unique<HGCeeUncalibratedRecHitCollection>();
  auto hefUncalibRechits = std::make_unique<HGChefUncalibratedRecHitCollection>();
  auto hebUncalibRechits = std::make_unique<HGChebUncalibratedRecHitCollection>();
  auto hfnoseUncalibRechits = std::make_unique<HGChfnoseUncalibratedRecHitCollection>();

  // loop over HGCEE digis
  const auto& pHGCEEDigis = evt.getHandle(eeDigiCollection_);
  if (pHGCEEDigis.isValid())
    worker_->runHGCEE(es.getHandle(ee_geometry_token_), *pHGCEEDigis, *eeUncalibRechits);

  // loop over HGCHEsil digis
  const auto& pHGCHEFDigis = evt.getHandle(hefDigiCollection_);
  if (pHGCHEFDigis.isValid())
    worker_->runHGCHEsil(es.getHandle(hef_geometry_token_), *pHGCHEFDigis, *hefUncalibRechits);

  // loop over HGCHEscint digis
  const auto& pHGCHEBDigis = evt.getHandle(hebDigiCollection_);
  if (pHGCHEBDigis.isValid())
    worker_->runHGCHEscint(es.getHandle(heb_geometry_token_), *pHGCHEBDigis, *hebUncalibRechits);

  // loop over HFNose digis
  const auto& pHGCHFNoseDigis = evt.getHandle(hfnoseDigiCollection_);
  if (pHGCHFNoseDigis.isValid())
    worker_->runHGCHFNose(es.getHandle(hfnose_geometry_token_), *pHGCHFNoseDigis, *hfnoseUncalibRechits);

  // put the collection of recunstructed hits in the event
  evt.put(std::move(eeUncalibRechits), eeHitCollection_);
  evt.put(std::move(hefUncalibRechits), hefHitCollection_);
  evt.put(std::move(hebUncalibRechits), hebHitCollection_);
  if (pHGCHFNoseDigis.isValid())
    evt.put(std::move(hfnoseUncalibRechits), hfnoseHitCollection_);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HGCalUncalibRecHitProducer);
