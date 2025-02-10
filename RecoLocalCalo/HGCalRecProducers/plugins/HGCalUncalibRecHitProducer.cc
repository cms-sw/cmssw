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
          ps.getParameter<std::string>("algo"), ps, consumesCollector(), ps.getParameter<bool>("computeLocalTime"))} {
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
  if (pHGCEEDigis.isValid()) {
    worker_->runHGCEE(es.getHandle(ee_geometry_token_), *pHGCEEDigis, *eeUncalibRechits);
    evt.put(std::move(eeUncalibRechits), eeHitCollection_);
  }
  // loop over HGCHEsil digis
  const auto& pHGCHEFDigis = evt.getHandle(hefDigiCollection_);
  if (pHGCHEFDigis.isValid()) {
    worker_->runHGCHEsil(es.getHandle(hef_geometry_token_), *pHGCHEFDigis, *hefUncalibRechits);
    evt.put(std::move(hefUncalibRechits), hefHitCollection_);
  }
  // loop over HGCHEscint digis
  const auto& pHGCHEBDigis = evt.getHandle(hebDigiCollection_);
  if (pHGCHEBDigis.isValid()) {
    worker_->runHGCHEscint(es.getHandle(heb_geometry_token_), *pHGCHEBDigis, *hebUncalibRechits);
    evt.put(std::move(hebUncalibRechits), hebHitCollection_);
  }
  // loop over HFNose digis
  const auto& pHGCHFNoseDigis = evt.getHandle(hfnoseDigiCollection_);
  if (pHGCHFNoseDigis.isValid()) {
    worker_->runHGCHFNose(es.getHandle(hfnose_geometry_token_), *pHGCHFNoseDigis, *hfnoseUncalibRechits);
    evt.put(std::move(hfnoseUncalibRechits), hfnoseHitCollection_);
  }
}

void HGCalUncalibRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // HGCalUncalibRecHit
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("HGCEEdigiCollection", edm::InputTag("hgcalDigis", "EE"));
  desc.add<std::string>("HGCEEhitCollection", "HGCEEUncalibRecHits");
  desc.add<edm::InputTag>("HGCHEFdigiCollection", edm::InputTag("hgcalDigis", "HEfront"));
  desc.add<std::string>("HGCHEFhitCollection", "HGCHEFUncalibRecHits");
  desc.add<edm::InputTag>("HGCHEBdigiCollection", edm::InputTag("hgcalDigis", "HEback"));
  desc.add<std::string>("HGCHEBhitCollection", "HGCHEBUncalibRecHits");
  desc.add<edm::InputTag>("HGCHFNosedigiCollection", edm::InputTag("hfnoseDigis", "HFNose"));
  desc.add<std::string>("HGCHFNosehitCollection", "HGCHFNoseUncalibRecHits");
  edm::ParameterSetDescription HGCEEConfigPSet;
  HGCEEConfigPSet.add<bool>("isSiFE", true);
  HGCEEConfigPSet.add<unsigned int>("adcNbits", 10);
  HGCEEConfigPSet.add<double>("adcSaturation", 100);
  HGCEEConfigPSet.add<unsigned int>("tdcNbits", 12);
  HGCEEConfigPSet.add<double>("tdcSaturation", 10000);
  HGCEEConfigPSet.add<double>("tdcOnset", 60);
  HGCEEConfigPSet.add<double>("toaLSB_ns", 0.0244);
  HGCEEConfigPSet.add<double>("tofDelay", -9);
  HGCEEConfigPSet.add<std::vector<double>>("fCPerMIP",
                                           {
                                               1.25,
                                               2.57,
                                               3.88,
                                           });
  desc.add<edm::ParameterSetDescription>("HGCEEConfig", HGCEEConfigPSet);
  edm::ParameterSetDescription HGCHEFConfigPSet;
  HGCHEFConfigPSet.add<bool>("isSiFE", true);
  HGCHEFConfigPSet.add<unsigned int>("adcNbits", 10);
  HGCHEFConfigPSet.add<double>("adcSaturation", 100);
  HGCHEFConfigPSet.add<unsigned int>("tdcNbits", 12);
  HGCHEFConfigPSet.add<double>("tdcSaturation", 10000);
  HGCHEFConfigPSet.add<double>("tdcOnset", 60);
  HGCHEFConfigPSet.add<double>("toaLSB_ns", 0.0244);
  HGCHEFConfigPSet.add<double>("tofDelay", -11);
  HGCHEFConfigPSet.add<std::vector<double>>("fCPerMIP",
                                            {
                                                1.25,
                                                2.57,
                                                3.88,
                                            });
  desc.add<edm::ParameterSetDescription>("HGCHEFConfig", HGCHEFConfigPSet);
  edm::ParameterSetDescription HGCHEBConfigPSet;
  HGCHEBConfigPSet.add<bool>("isSiFE", true);
  HGCHEBConfigPSet.add<unsigned int>("adcNbits", 10);
  HGCHEBConfigPSet.add<double>("adcSaturation", 68.75);
  HGCHEBConfigPSet.add<unsigned int>("tdcNbits", 12);
  HGCHEBConfigPSet.add<double>("tdcSaturation", 1000);
  HGCHEBConfigPSet.add<double>("tdcOnset", 55);
  HGCHEBConfigPSet.add<double>("toaLSB_ns", 0.0244);
  HGCHEBConfigPSet.add<double>("tofDelay", -14);
  HGCHEBConfigPSet.add<std::vector<double>>("fCPerMIP",
                                            {
                                                1.0,
                                                1.0,
                                                1.0,
                                            });
  desc.add<edm::ParameterSetDescription>("HGCHEBConfig", HGCHEBConfigPSet);
  edm::ParameterSetDescription HGCHFNoseConfigPSet;
  HGCHFNoseConfigPSet.add<bool>("isSiFE", false);
  HGCHFNoseConfigPSet.add<unsigned int>("adcNbits", 10);
  HGCHFNoseConfigPSet.add<double>("adcSaturation", 100);
  HGCHFNoseConfigPSet.add<unsigned int>("tdcNbits", 12);
  HGCHFNoseConfigPSet.add<double>("tdcSaturation", 10000);
  HGCHFNoseConfigPSet.add<double>("tdcOnset", 60);
  HGCHFNoseConfigPSet.add<double>("toaLSB_ns", 0.0244);
  HGCHFNoseConfigPSet.add<double>("tofDelay", -33);
  HGCHFNoseConfigPSet.add<std::vector<double>>("fCPerMIP",
                                               {
                                                   1.25,
                                                   2.57,
                                                   3.88,
                                               });
  desc.add<edm::ParameterSetDescription>("HGCHFNoseConfig", HGCHFNoseConfigPSet);
  desc.add<std::string>("algo", "HGCalUncalibRecHitWorkerWeights");
  desc.add<bool>("computeLocalTime", false);
  descriptions.add("HGCalUncalibRecHitProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HGCalUncalibRecHitProducer);
