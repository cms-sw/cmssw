#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Math/interface/libminifloat.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/Scouting/interface/Run3ScoutingEBRecHit.h"
#include "DataFormats/Scouting/interface/Run3ScoutingEERecHit.h"
#include "DataFormats/Scouting/interface/Run3ScoutingHBHERecHit.h"

class HLTScoutingRecHitProducer : public edm::global::EDProducer<> {
public:
  explicit HLTScoutingRecHitProducer(const edm::ParameterSet&);
  ~HLTScoutingRecHitProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const final;
  static void produceEcal(edm::Event& iEvent,
                          const reco::PFRecHitCollection& inputRecHits,
                          double minEnergyEB,
                          double minEnergyEE,
                          int mantissaPrecision,
                          const std::string& tag = "");
  static void produceHcal(edm::Event& iEvent,
                          const reco::PFRecHitCollection& inputRecHits,
                          double minEnergyHBHE,
                          int mantissaPrecision,
                          const std::string& tag = "");

  template <typename T>
  void setToken(edm::EDGetTokenT<T>& token, const edm::ParameterSet& iConfig, std::string name) {
    const auto inputTag = iConfig.getParameter<edm::InputTag>(name);
    if (!inputTag.encode().empty()) {
      token = consumes(inputTag);
    }
  }

  edm::EDGetTokenT<reco::PFRecHitCollection> recoPFRecHitsTokenECAL_;
  edm::EDGetTokenT<reco::PFRecHitCollection> recoPFRecHitsTokenECALCleaned_;
  edm::EDGetTokenT<reco::PFRecHitCollection> recoPFRecHitsTokenHBHE_;
  const double minEnergyEB_;
  const double minEnergyEE_;
  const double minEnergyCleanedEB_;
  const double minEnergyCleanedEE_;
  const double minEnergyHBHE_;
  const int mantissaPrecision_;
};

HLTScoutingRecHitProducer::HLTScoutingRecHitProducer(const edm::ParameterSet& iConfig)
    : minEnergyEB_(iConfig.getParameter<double>("minEnergyEB")),
      minEnergyEE_(iConfig.getParameter<double>("minEnergyEE")),
      minEnergyCleanedEB_(iConfig.getParameter<double>("minEnergyCleanedEB")),
      minEnergyCleanedEE_(iConfig.getParameter<double>("minEnergyCleanedEE")),
      minEnergyHBHE_(iConfig.getParameter<double>("minEnergyHBHE")),
      mantissaPrecision_(iConfig.getParameter<int>("mantissaPrecision")) {
  //this is done this way so that if an empty InputTag is provided, the token will be set to an uninitialized state and we'll skip processing that product
  //this protects against types pfRecItsHBHE  is a type and we dont want to slightly pass
  setToken(recoPFRecHitsTokenECAL_, iConfig, "pfRecHitsECAL");
  setToken(recoPFRecHitsTokenECALCleaned_, iConfig, "pfRecHitsECALCleaned");
  setToken(recoPFRecHitsTokenHBHE_, iConfig, "pfRecHitsHBHE");
  produces<Run3ScoutingEBRecHitCollection>("EB");
  produces<Run3ScoutingEERecHitCollection>("EE");
  produces<Run3ScoutingEBRecHitCollection>("EBCleaned");
  produces<Run3ScoutingEERecHitCollection>("EECleaned");
  produces<Run3ScoutingHBHERecHitCollection>("HBHE");
}

void HLTScoutingRecHitProducer::produceEcal(edm::Event& iEvent,
                                            const reco::PFRecHitCollection& inputRecHits,
                                            double minEnergyEB,
                                            double minEnergyEE,
                                            int mantissaPrecision,
                                            const std::string& tag) {
  auto run3ScoutEBRecHits = std::make_unique<Run3ScoutingEBRecHitCollection>();
  run3ScoutEBRecHits->reserve(inputRecHits.size());

  auto run3ScoutEERecHits = std::make_unique<Run3ScoutingEERecHitCollection>();
  run3ScoutEERecHits->reserve(inputRecHits.size());

  for (auto const& rh : inputRecHits) {
    if (rh.layer() == PFLayer::ECAL_BARREL) {
      if (rh.energy() < minEnergyEB) {
        continue;
      }

      run3ScoutEBRecHits->emplace_back(
          MiniFloatConverter::reduceMantissaToNbitsRounding(rh.energy(), mantissaPrecision),
          MiniFloatConverter::reduceMantissaToNbitsRounding(rh.time(), mantissaPrecision),
          rh.detId(),
          rh.flags());
    } else if (rh.layer() == PFLayer::ECAL_ENDCAP) {
      if (rh.energy() < minEnergyEE) {
        continue;
      }

      run3ScoutEERecHits->emplace_back(
          MiniFloatConverter::reduceMantissaToNbitsRounding(rh.energy(), mantissaPrecision),
          MiniFloatConverter::reduceMantissaToNbitsRounding(rh.time(), mantissaPrecision),
          rh.detId());
    } else {
      edm::LogWarning("HLTScoutingRecHitProducer")
          << "Skipping PFRecHit because of unexpected PFLayer value (" << rh.layer() << ").";
    }
  }
  iEvent.put(std::move(run3ScoutEBRecHits), "EB" + tag);
  iEvent.put(std::move(run3ScoutEERecHits), "EE" + tag);
}

void HLTScoutingRecHitProducer::produceHcal(edm::Event& iEvent,
                                            const reco::PFRecHitCollection& inputRecHits,
                                            double minEnergyHBHE,
                                            int mantissaPrecision,
                                            const std::string& tag) {
  auto run3ScoutHBHERecHits = std::make_unique<Run3ScoutingHBHERecHitCollection>();
  run3ScoutHBHERecHits->reserve(inputRecHits.size());

  for (auto const& rh : inputRecHits) {
    if (rh.energy() < minEnergyHBHE) {
      continue;
    }

    run3ScoutHBHERecHits->emplace_back(
        MiniFloatConverter::reduceMantissaToNbitsRounding(rh.energy(), mantissaPrecision),
        MiniFloatConverter::reduceMantissaToNbitsRounding(rh.time(), mantissaPrecision),
        rh.detId());
  }

  iEvent.put(std::move(run3ScoutHBHERecHits), "HBHE" + tag);
}

void HLTScoutingRecHitProducer::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  // ECAL
  if (!recoPFRecHitsTokenECAL_.isUninitialized()) {
    auto const& recoPFRecHitsECAL = iEvent.get(recoPFRecHitsTokenECAL_);
    produceEcal(iEvent, recoPFRecHitsECAL, minEnergyEB_, minEnergyEE_, mantissaPrecision_);
  }
  // Cleaned ECAL
  if (!recoPFRecHitsTokenECALCleaned_.isUninitialized()) {
    auto const& recoPFRecHitsECALCleaned = iEvent.get(recoPFRecHitsTokenECALCleaned_);
    produceEcal(
        iEvent, recoPFRecHitsECALCleaned, minEnergyCleanedEB_, minEnergyCleanedEE_, mantissaPrecision_, "Cleaned");
  }
  // HBHE
  if (!recoPFRecHitsTokenHBHE_.isUninitialized()) {
    auto const& recoPFRecHitsHBHE = iEvent.get(recoPFRecHitsTokenHBHE_);
    produceHcal(iEvent, recoPFRecHitsHBHE, minEnergyHBHE_, mantissaPrecision_);
  }
}

void HLTScoutingRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pfRecHitsECAL", edm::InputTag("hltParticleFlowRecHitECALUnseeded"));
  desc.add<edm::InputTag>("pfRecHitsECALCleaned", edm::InputTag("hltParticleFlowRecHitECALUnseeded", "Cleaned"));
  desc.add<edm::InputTag>("pfRecHitsHBHE", edm::InputTag("hltParticleFlowRecHitHBHE"));
  desc.add<double>("minEnergyEB", -1)->setComment("Minimum energy of the EcalBarrel PFRecHit in GeV");
  desc.add<double>("minEnergyEE", -1)->setComment("Minimum energy of the EcalEndcap PFRecHit in GeV");
  desc.add<double>("minEnergyCleanedEB", -1)->setComment("Minimum energy of the cleaned EcalBarrel PFRecHit in GeV");
  desc.add<double>("minEnergyCleanedEE", -1)->setComment("Minimum energy of the cleaned EcalEndcap PFRecHit in GeV");
  desc.add<double>("minEnergyHBHE", -1)->setComment("Minimum energy of the HBHE PFRecHit in GeV");
  desc.add<int>("mantissaPrecision", 10)->setComment("default of 10 corresponds to float16, change to 23 for float32");
  descriptions.add("hltScoutingRecHitProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTScoutingRecHitProducer);
