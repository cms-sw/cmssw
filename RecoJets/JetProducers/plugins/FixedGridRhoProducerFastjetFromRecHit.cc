/*

Author: Swagata Mukherjee

Date: November 2021

This producer computes rho from ECAL and HCAL recHits. 
The current plan is to use it in egamma and muon HLT, who currently use 
the other producer called FixedGridRhoProducerFastjet.
At HLT, FixedGridRhoProducerFastjet takes calotowers as input and computes rho.
But calotowers are expected to be phased out sometime in Run3.
So this recHit-based rho producer, FixedGridRhoProducerFastjetFromRecHit, can be used as an alternative.

*/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include "fastjet/tools/GridMedianBackgroundEstimator.hh"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"
#include "CondTools/Hcal/interface/HcalPFCutsHandler.h"

class FixedGridRhoProducerFastjetFromRecHit : public edm::stream::EDProducer<> {
public:
  explicit FixedGridRhoProducerFastjetFromRecHit(const edm::ParameterSet &iConfig);
  ~FixedGridRhoProducerFastjetFromRecHit() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  std::array<double, 4> getHitP4(const DetId &detId, const double hitE, const CaloGeometry &caloGeometry) const;
  bool passedHcalNoiseCut(const HBHERecHit &hit, const HcalPFCuts *) const;
  bool passedEcalNoiseCut(const EcalRecHit &hit, const EcalPFRecHitThresholds &thresholds) const;

  fastjet::GridMedianBackgroundEstimator bge_;
  const edm::EDGetTokenT<HBHERecHitCollection> hbheRecHitsTag_;
  const edm::EDGetTokenT<EcalRecHitCollection> ebRecHitsTag_;
  const edm::EDGetTokenT<EcalRecHitCollection> eeRecHitsTag_;

  const EgammaHcalIsolation::arrayHB eThresHB_;
  const EgammaHcalIsolation::arrayHE eThresHE_;

  //Muon HLT currently use ECAL-only rho for ECAL isolation,
  //and HCAL-only rho for HCAL isolation. So, this skipping option is needed.
  const bool skipHCAL_;
  const bool skipECAL_;

  const edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd> ecalPFRecHitThresholdsToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;

  // following are needed to grab HCal thresholds from GT
  edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd> hcalCutsToken_;
  const bool cutsFromDB_;
  HcalPFCuts const *paramPF_ = nullptr;
};

FixedGridRhoProducerFastjetFromRecHit::FixedGridRhoProducerFastjetFromRecHit(const edm::ParameterSet &iConfig)
    : bge_(iConfig.getParameter<double>("maxRapidity"), iConfig.getParameter<double>("gridSpacing")),
      hbheRecHitsTag_(consumes(iConfig.getParameter<edm::InputTag>("hbheRecHitsTag"))),
      ebRecHitsTag_(consumes(iConfig.getParameter<edm::InputTag>("ebRecHitsTag"))),
      eeRecHitsTag_(consumes(iConfig.getParameter<edm::InputTag>("eeRecHitsTag"))),
      eThresHB_(iConfig.getParameter<EgammaHcalIsolation::arrayHB>("eThresHB")),
      eThresHE_(iConfig.getParameter<EgammaHcalIsolation::arrayHE>("eThresHE")),
      skipHCAL_(iConfig.getParameter<bool>("skipHCAL")),
      skipECAL_(iConfig.getParameter<bool>("skipECAL")),
      ecalPFRecHitThresholdsToken_{esConsumes()},
      caloGeometryToken_{esConsumes()},
      cutsFromDB_(iConfig.getParameter<bool>("usePFThresholdsFromDB")) {
  if (skipHCAL_ && skipECAL_) {
    throw cms::Exception("FixedGridRhoProducerFastjetFromRecHit")
        << "skipHCAL and skipECAL both can't be True. Please make at least one of them False.";
  }
  if (cutsFromDB_) {
    hcalCutsToken_ = esConsumes<HcalPFCuts, HcalPFCutsRcd, edm::Transition::BeginRun>(edm::ESInputTag("", "withTopo"));
  }
  produces<double>();
}

void FixedGridRhoProducerFastjetFromRecHit::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  //We use raw recHits and not the PF recHits, because in Phase1 muon and egamma HLT,
  //PF recHit collection is regional, while the full raw recHit collection is available.
  desc.add<edm::InputTag>("hbheRecHitsTag", edm::InputTag("hltHbhereco"));
  desc.add<edm::InputTag>("ebRecHitsTag", edm::InputTag("hltEcalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("eeRecHitsTag", edm::InputTag("hltEcalRecHit", "EcalRecHitsEE"));
  desc.add<bool>("skipHCAL", false);
  desc.add<bool>("skipECAL", false);
  //eThresHB/HE are from RecoParticleFlow/PFClusterProducer/python/particleFlowRecHitHBHE_cfi.py
  desc.add<std::vector<double> >("eThresHB", {0.1, 0.2, 0.3, 0.3});
  desc.add<std::vector<double> >("eThresHE", {0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
  desc.add<double>("maxRapidity", 2.5);
  desc.add<double>("gridSpacing", 0.55);
  desc.add<bool>("usePFThresholdsFromDB", true);
  descriptions.addWithDefaultLabel(desc);
}

FixedGridRhoProducerFastjetFromRecHit::~FixedGridRhoProducerFastjetFromRecHit() = default;

void FixedGridRhoProducerFastjetFromRecHit::beginRun(edm::Run const &r, edm::EventSetup const &es) {
  if (cutsFromDB_) {
    paramPF_ = &es.getData(hcalCutsToken_);
  }
}

void FixedGridRhoProducerFastjetFromRecHit::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  std::vector<fastjet::PseudoJet> inputs;
  auto const &thresholds = iSetup.getData(ecalPFRecHitThresholdsToken_);
  auto const &caloGeometry = iSetup.getData(caloGeometryToken_);

  if (!skipHCAL_) {
    auto const &hbheRecHits = iEvent.get(hbheRecHitsTag_);
    inputs.reserve(inputs.size() + hbheRecHits.size());
    for (const auto &hit : hbheRecHits) {
      if (passedHcalNoiseCut(hit, paramPF_)) {
        const auto &hitp4 = getHitP4(hit.id(), hit.energy(), caloGeometry);
        inputs.emplace_back(fastjet::PseudoJet(hitp4[0], hitp4[1], hitp4[2], hitp4[3]));
      }
    }
  }

  if (!skipECAL_) {
    auto const &ebRecHits = iEvent.get(ebRecHitsTag_);
    inputs.reserve(inputs.size() + ebRecHits.size());
    for (const auto &hit : ebRecHits) {
      if (passedEcalNoiseCut(hit, thresholds)) {
        const auto &hitp4 = getHitP4(hit.id(), hit.energy(), caloGeometry);
        inputs.emplace_back(fastjet::PseudoJet(hitp4[0], hitp4[1], hitp4[2], hitp4[3]));
      }
    }

    auto const &eeRecHits = iEvent.get(eeRecHitsTag_);
    inputs.reserve(inputs.size() + eeRecHits.size());
    for (const auto &hit : eeRecHits) {
      if (passedEcalNoiseCut(hit, thresholds)) {
        const auto &hitp4 = getHitP4(hit.id(), hit.energy(), caloGeometry);
        inputs.emplace_back(fastjet::PseudoJet(hitp4[0], hitp4[1], hitp4[2], hitp4[3]));
      }
    }
  }

  bge_.set_particles(inputs);
  iEvent.put(std::make_unique<double>(bge_.rho()));
}

std::array<double, 4> FixedGridRhoProducerFastjetFromRecHit::getHitP4(const DetId &detId,
                                                                      const double hitE,
                                                                      const CaloGeometry &caloGeometry) const {
  const CaloSubdetectorGeometry *subDetGeom = caloGeometry.getSubdetectorGeometry(detId);
  const auto &gpPos = subDetGeom->getGeometry(detId)->repPos();
  const double thispt = hitE / cosh(gpPos.eta());
  const double thispx = thispt * cos(gpPos.phi());
  const double thispy = thispt * sin(gpPos.phi());
  const double thispz = thispt * sinh(gpPos.eta());
  return std::array<double, 4>{{thispx, thispy, thispz, hitE}};
}

//HCAL noise cleaning cuts.
bool FixedGridRhoProducerFastjetFromRecHit::passedHcalNoiseCut(const HBHERecHit &hit,
                                                               const HcalPFCuts *hcalcuts) const {
  if (hcalcuts != nullptr) {  // using hcal cuts from DB
    const HcalPFCut *item = hcalcuts->getValues(hit.id().rawId());
    return (hit.energy() > item->noiseThreshold());
  } else {  // using hcal cuts from config file
    const auto thisDetId = hit.id();
    const auto thisDepth = thisDetId.depth();
    return (thisDetId.subdet() == HcalBarrel && hit.energy() > eThresHB_[thisDepth - 1]) ||
           (thisDetId.subdet() == HcalEndcap && hit.energy() > eThresHE_[thisDepth - 1]);
  }
}

//ECAL noise cleaning cuts using per-crystal PF-recHit thresholds.
bool FixedGridRhoProducerFastjetFromRecHit::passedEcalNoiseCut(const EcalRecHit &hit,
                                                               const EcalPFRecHitThresholds &thresholds) const {
  return (hit.energy() > thresholds[hit.detid()]);
}

DEFINE_FWK_MODULE(FixedGridRhoProducerFastjetFromRecHit);
