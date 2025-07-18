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

  const edm::EDGetTokenT<reco::PFRecHitCollection> recoPFRecHitsTokenECAL_;
  const edm::EDGetTokenT<reco::PFRecHitCollection> recoPFRecHitsTokenECALCleaned_;
  const edm::EDGetTokenT<reco::PFRecHitCollection> recoPFRecHitsTokenHBHE_;
  const double minEnergyEB_;
  const double minEnergyEE_;
  const double minEnergyCleanedEB_;
  const double minEnergyCleanedEE_;
  const double minEnergyHBHE_;
  const int mantissaPrecision_;
};

HLTScoutingRecHitProducer::HLTScoutingRecHitProducer(const edm::ParameterSet& iConfig)
    : recoPFRecHitsTokenECAL_(consumes(iConfig.getParameter<edm::InputTag>("pfRecHitsECAL"))),
      recoPFRecHitsTokenECALCleaned_(consumes(iConfig.getParameter<edm::InputTag>("pfRecHitsECALCleaned"))),      
      recoPFRecHitsTokenHBHE_(consumes(iConfig.getParameter<edm::InputTag>("pfRecHitsHBHE"))),
      minEnergyEB_(iConfig.getParameter<double>("minEnergyEB")),
      minEnergyEE_(iConfig.getParameter<double>("minEnergyEE")),
      minEnergyCleanedEB_(iConfig.getParameter<double>("minEnergyCleanedEB")),
      minEnergyCleanedEE_(iConfig.getParameter<double>("minEnergyCleanedEE")),
      minEnergyHBHE_(iConfig.getParameter<double>("minEnergyHBHE")),
      mantissaPrecision_(iConfig.getParameter<int>("mantissaPrecision")) {
  produces<Run3ScoutingEBRecHitCollection>("EB");
  produces<Run3ScoutingEERecHitCollection>("EE");
  produces<Run3ScoutingEBRecHitCollection>("EBCleaned");
  produces<Run3ScoutingEERecHitCollection>("EECleaned");
  produces<Run3ScoutingHBHERecHitCollection>("HBHE");
}

void HLTScoutingRecHitProducer::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  // ECAL
  auto const& recoPFRecHitsECAL = iEvent.get(recoPFRecHitsTokenECAL_);

  auto run3ScoutEBRecHits = std::make_unique<Run3ScoutingEBRecHitCollection>();
  run3ScoutEBRecHits->reserve(recoPFRecHitsECAL.size());

  auto run3ScoutEERecHits = std::make_unique<Run3ScoutingEERecHitCollection>();
  run3ScoutEERecHits->reserve(recoPFRecHitsECAL.size());

  for (auto const& rh : recoPFRecHitsECAL) {
    if (rh.layer() == PFLayer::ECAL_BARREL) {
      if (rh.energy() < minEnergyEB_) {
        continue;
      }

      run3ScoutEBRecHits->emplace_back(
          MiniFloatConverter::reduceMantissaToNbitsRounding(rh.energy(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(rh.time(), mantissaPrecision_),
          rh.detId(),
          rh.flags());
    } else if (rh.layer() == PFLayer::ECAL_ENDCAP) {
      if (rh.energy() < minEnergyEE_) {
        continue;
      }

      run3ScoutEERecHits->emplace_back(
          MiniFloatConverter::reduceMantissaToNbitsRounding(rh.energy(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(rh.time(), mantissaPrecision_),
          rh.detId());
    } else {
      edm::LogWarning("HLTScoutingRecHitProducer")
          << "Skipping PFRecHit because of unexpected PFLayer value (" << rh.layer() << ").";
    }
  }

  iEvent.put(std::move(run3ScoutEBRecHits), "EB");
  iEvent.put(std::move(run3ScoutEERecHits), "EE");

  // Cleaned ECAL
  auto const& recoPFRecHitsECALCleaned = iEvent.get(recoPFRecHitsTokenECALCleaned_);
  auto run3ScoutEBRecHitsCleaned = std::make_unique<Run3ScoutingEBRecHitCollection>();
  run3ScoutEBRecHitsCleaned->reserve(recoPFRecHitsECALCleaned.size());
  auto run3ScoutEERecHitsCleaned = std::make_unique<Run3ScoutingEERecHitCollection>();
  run3ScoutEERecHitsCleaned->reserve(recoPFRecHitsECALCleaned.size());
  for (auto const& rh : recoPFRecHitsECALCleaned) {
    if (rh.layer() == PFLayer::ECAL_BARREL) {
      if (rh.energy() < minEnergyCleanedEB_) {
        continue;
      }
      run3ScoutEBRecHitsCleaned->emplace_back(
          MiniFloatConverter::reduceMantissaToNbitsRounding(rh.energy(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(rh.time(), mantissaPrecision_),
          rh.detId(),
          rh.flags());
    } else if (rh.layer() == PFLayer::ECAL_ENDCAP) {
      if (rh.energy() < minEnergyCleanedEE_) {
        continue;
      }
      run3ScoutEERecHitsCleaned->emplace_back(
          MiniFloatConverter::reduceMantissaToNbitsRounding(rh.energy(), mantissaPrecision_),
          MiniFloatConverter::reduceMantissaToNbitsRounding(rh.time(), mantissaPrecision_),
          rh.detId());
    }
  }
  iEvent.put(std::move(run3ScoutEBRecHitsCleaned), "EBCleaned");
  iEvent.put(std::move(run3ScoutEERecHitsCleaned), "EECleaned");

  // HBHE
  auto const& recoPFRecHitsHBHE = iEvent.get(recoPFRecHitsTokenHBHE_);

  auto run3ScoutHBHERecHits = std::make_unique<Run3ScoutingHBHERecHitCollection>();
  run3ScoutHBHERecHits->reserve(recoPFRecHitsHBHE.size());

  for (auto const& rh : recoPFRecHitsHBHE) {
    if (rh.energy() < minEnergyHBHE_) {
      continue;
    }

    run3ScoutHBHERecHits->emplace_back(
        MiniFloatConverter::reduceMantissaToNbitsRounding(rh.energy(), mantissaPrecision_), rh.detId());
  }

  iEvent.put(std::move(run3ScoutHBHERecHits), "HBHE");
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
