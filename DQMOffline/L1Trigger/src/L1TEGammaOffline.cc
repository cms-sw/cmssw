#include "DQMOffline/L1Trigger/interface/L1TEGammaOffline.h"
#include "DQMOffline/L1Trigger/interface/L1TFillWithinLimits.h"
#include "DQMOffline/L1Trigger/interface/L1TCommon.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
// Geometry
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "TLorentzVector.h"

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>

const std::map<std::string, unsigned int> L1TEGammaOffline::PlotConfigNames = {
    {"nVertex", PlotConfig::nVertex}, {"ETvsET", PlotConfig::ETvsET}, {"PHIvsPHI", PlotConfig::PHIvsPHI}};

//
// -------------------------------------- Constructor --------------------------------------------
//
L1TEGammaOffline::L1TEGammaOffline(const edm::ParameterSet& ps)
    : theGsfElectronCollection_(
          consumes<reco::GsfElectronCollection>(ps.getParameter<edm::InputTag>("electronCollection"))),
      thePhotonCollection_(consumes<std::vector<reco::Photon> >(ps.getParameter<edm::InputTag>("photonCollection"))),
      thePVCollection_(consumes<reco::VertexCollection>(ps.getParameter<edm::InputTag>("PVCollection"))),
      theBSCollection_(consumes<reco::BeamSpot>(ps.getParameter<edm::InputTag>("beamSpotCollection"))),
      triggerInputTag_(consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("triggerInputTag"))),
      triggerResultsInputTag_(consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("triggerResults"))),
      triggerProcess_(ps.getParameter<std::string>("triggerProcess")),
      triggerNames_(ps.getParameter<std::vector<std::string> >("triggerNames")),
      histFolder_(ps.getParameter<std::string>("histFolder")),
      efficiencyFolder_(histFolder_ + "/efficiency_raw"),
      stage2CaloLayer2EGammaToken_(
          consumes<l1t::EGammaBxCollection>(ps.getParameter<edm::InputTag>("stage2CaloLayer2EGammaSource"))),
      electronEfficiencyThresholds_(ps.getParameter<std::vector<double> >("electronEfficiencyThresholds")),
      electronEfficiencyBins_(ps.getParameter<std::vector<double> >("electronEfficiencyBins")),
      probeToL1Offset_(ps.getParameter<double>("probeToL1Offset")),
      deepInspectionElectronThresholds_(ps.getParameter<std::vector<double> >("deepInspectionElectronThresholds")),
      photonEfficiencyThresholds_(ps.getParameter<std::vector<double> >("photonEfficiencyThresholds")),
      photonEfficiencyBins_(ps.getParameter<std::vector<double> >("photonEfficiencyBins")),
      maxDeltaRForL1Matching_(ps.getParameter<double>("maxDeltaRForL1Matching")),
      maxDeltaRForHLTMatching_(ps.getParameter<double>("maxDeltaRForHLTMatching")),
      recoToL1TThresholdFactor_(ps.getParameter<double>("recoToL1TThresholdFactor")),
      tagElectron_(),
      probeElectron_(),
      tagAndProbleInvariantMass_(-1.),
      hltConfig_(),
      triggerIndices_(),
      triggerResults_(),
      triggerEvent_(),
      histDefinitions_(dqmoffline::l1t::readHistDefinitions(ps.getParameterSet("histDefinitions"), PlotConfigNames)),
      h_nVertex_(),
      h_tagAndProbeMass_(),
      h_L1EGammaETvsElectronET_EB_(),
      h_L1EGammaETvsElectronET_EE_(),
      h_L1EGammaETvsElectronET_EB_EE_(),
      h_L1EGammaPhivsElectronPhi_EB_(),
      h_L1EGammaPhivsElectronPhi_EE_(),
      h_L1EGammaPhivsElectronPhi_EB_EE_(),
      h_L1EGammaEtavsElectronEta_(),
      h_resolutionElectronET_EB_(),
      h_resolutionElectronET_EE_(),
      h_resolutionElectronET_EB_EE_(),
      h_resolutionElectronPhi_EB_(),
      h_resolutionElectronPhi_EE_(),
      h_resolutionElectronPhi_EB_EE_(),
      h_resolutionElectronEta_(),
      h_efficiencyElectronET_EB_pass_(),
      h_efficiencyElectronET_EE_pass_(),
      h_efficiencyElectronET_EB_EE_pass_(),
      h_efficiencyElectronPhi_vs_Eta_pass_(),
      h_efficiencyElectronEta_pass_(),
      h_efficiencyElectronPhi_pass_(),
      h_efficiencyElectronNVertex_pass_(),
      h_efficiencyElectronET_EB_total_(),
      h_efficiencyElectronET_EE_total_(),
      h_efficiencyElectronET_EB_EE_total_(),
      h_efficiencyElectronPhi_vs_Eta_total_(),
      h_efficiencyElectronEta_total_(),
      h_efficiencyElectronPhi_total_(),
      h_efficiencyElectronNVertex_total_(),
      h_L1EGammaETvsPhotonET_EB_(),
      h_L1EGammaETvsPhotonET_EE_(),
      h_L1EGammaETvsPhotonET_EB_EE_(),
      h_L1EGammaPhivsPhotonPhi_EB_(),
      h_L1EGammaPhivsPhotonPhi_EE_(),
      h_L1EGammaPhivsPhotonPhi_EB_EE_(),
      h_L1EGammaEtavsPhotonEta_(),
      h_resolutionPhotonEta_(),
      h_efficiencyPhotonET_EB_pass_(),
      h_efficiencyPhotonET_EE_pass_(),
      h_efficiencyPhotonET_EB_EE_pass_(),
      h_efficiencyPhotonET_EB_total_(),
      h_efficiencyPhotonET_EE_total_(),
      h_efficiencyPhotonET_EB_EE_total_() {
  edm::LogInfo("L1TEGammaOffline") << "Constructor "
                                   << "L1TEGammaOffline::L1TEGammaOffline " << std::endl;
}

//
// -- Destructor
//
L1TEGammaOffline::~L1TEGammaOffline() {
  edm::LogInfo("L1TEGammaOffline") << "Destructor L1TEGammaOffline::~L1TEGammaOffline " << std::endl;
}

//
// -------------------------------------- beginRun --------------------------------------------
//
void L1TEGammaOffline::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::LogInfo("L1TEGammaOffline") << "L1TEGammaOffline::beginRun" << std::endl;
  bool changed(true);
  if (!hltConfig_.init(iRun, iSetup, triggerProcess_, changed)) {
    edm::LogError("L1TEGammaOffline") << " HLT config extraction failure with process name " << triggerProcess_
                                      << std::endl;
    triggerNames_.clear();
  } else {
    triggerIndices_ = dqmoffline::l1t::getTriggerIndices(triggerNames_, hltConfig_.triggerNames());
  }
}

//
// -------------------------------------- bookHistos --------------------------------------------
//
void L1TEGammaOffline::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  edm::LogInfo("L1TEGammaOffline") << "L1TEGammaOffline::bookHistograms" << std::endl;

  //book at beginRun
  bookElectronHistos(ibooker);
  // bookPhotonHistos(ibooker);
}

//
// -------------------------------------- Analyze --------------------------------------------
//
void L1TEGammaOffline::analyze(edm::Event const& e, edm::EventSetup const& eSetup) {
  edm::LogInfo("L1TEGammaOffline") << "L1TEGammaOffline::analyze" << std::endl;

  edm::Handle<edm::TriggerResults> triggerResultHandle;
  e.getByToken(triggerResultsInputTag_, triggerResultHandle);
  if (!triggerResultHandle.isValid()) {
    edm::LogWarning("L1TEGammaOffline") << "invalid edm::TriggerResults handle" << std::endl;
    return;
  }
  triggerResults_ = *triggerResultHandle;

  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  e.getByToken(triggerInputTag_, triggerEventHandle);
  if (!triggerEventHandle.isValid()) {
    edm::LogWarning("L1TEGammaOffline") << "invalid trigger::TriggerEvent handle" << std::endl;
    return;
  }
  triggerEvent_ = *triggerEventHandle;

  edm::Handle<reco::VertexCollection> vertexHandle;
  e.getByToken(thePVCollection_, vertexHandle);
  if (!vertexHandle.isValid()) {
    edm::LogWarning("L1TEGammaOffline") << "invalid collection: vertex " << std::endl;
    return;
  }

  unsigned int nVertex = vertexHandle->size();
  dqmoffline::l1t::fillWithinLimits(h_nVertex_, nVertex);

  if (!dqmoffline::l1t::passesAnyTriggerFromList(triggerIndices_, triggerResults_)) {
    return;
  }
  // L1T
  fillElectrons(e, nVertex);
  // fillPhotons(e, nVertex);
}

void L1TEGammaOffline::fillElectrons(edm::Event const& e, const unsigned int nVertex) {
  edm::Handle<l1t::EGammaBxCollection> l1EGamma;
  e.getByToken(stage2CaloLayer2EGammaToken_, l1EGamma);

  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  e.getByToken(theGsfElectronCollection_, gsfElectrons);

  if (!gsfElectrons.isValid()) {
    edm::LogWarning("L1TEGammaOffline") << "invalid collection: GSF electrons " << std::endl;
    return;
  }
  if (gsfElectrons->empty()) {
    LogDebug("L1TEGammaOffline") << "empty collection: GSF electrons " << std::endl;
    return;
  }
  if (!l1EGamma.isValid()) {
    edm::LogWarning("L1TEGammaOffline") << "invalid collection: L1 EGamma " << std::endl;
    return;
  }
  if (!findTagAndProbePair(gsfElectrons)) {
    LogDebug("L1TEGammaOffline") << "Could not find a tag & probe pair" << std::endl;
    return;  //continue to next event
  }

  auto probeElectron = probeElectron_;

  // find corresponding L1 EG
  double minDeltaR = maxDeltaRForL1Matching_;
  l1t::EGamma closestL1EGamma;
  bool foundMatch = false;

  int bunchCrossing = 0;
  for (auto egamma = l1EGamma->begin(bunchCrossing); egamma != l1EGamma->end(bunchCrossing); ++egamma) {
    double currentDeltaR = deltaR(egamma->eta(), egamma->phi(), probeElectron.eta(), probeElectron.phi());
    if (currentDeltaR > minDeltaR) {
      continue;
    } else {
      minDeltaR = currentDeltaR;
      closestL1EGamma = *egamma;
      foundMatch = true;
    }
  }

  if (!foundMatch) {
    LogDebug("L1TEGammaOffline") << "Could not find a matching L1 EGamma " << std::endl;
    return;
  }

  double recoEt = probeElectron.et();
  double recoEta = probeElectron.eta();
  double recoPhi = probeElectron.phi();

  double l1Et = closestL1EGamma.et();
  double l1Eta = closestL1EGamma.eta();
  double l1Phi = closestL1EGamma.phi();

  // if no reco value, relative resolution does not make sense -> sort to overflow
  double outOfBounds = 9999;
  double resolutionEt = recoEt > 0 ? (l1Et - recoEt) / recoEt : outOfBounds;
  double resolutionEta = l1Eta - recoEta;
  double resolutionPhi = reco::deltaPhi(l1Phi, recoPhi);

  using namespace dqmoffline::l1t;
  // eta
  fill2DWithinLimits(h_L1EGammaEtavsElectronEta_, recoEta, l1Eta);
  fillWithinLimits(h_resolutionElectronEta_, resolutionEta);

  // plots for deeper inspection
  for (auto threshold : deepInspectionElectronThresholds_) {
    if (recoEt > threshold * recoToL1TThresholdFactor_) {
      fillWithinLimits(h_efficiencyElectronEta_total_[threshold], recoEta);
      fillWithinLimits(h_efficiencyElectronPhi_total_[threshold], recoPhi);
      fillWithinLimits(h_efficiencyElectronNVertex_total_[threshold], nVertex);
      if (l1Et > threshold + probeToL1Offset_) {
        fillWithinLimits(h_efficiencyElectronEta_pass_[threshold], recoEta);
        fillWithinLimits(h_efficiencyElectronPhi_pass_[threshold], recoPhi);
        fillWithinLimits(h_efficiencyElectronNVertex_pass_[threshold], nVertex);
      }
    }
  }

  for (auto threshold : electronEfficiencyThresholds_) {
    if (recoEt > threshold * recoToL1TThresholdFactor_) {
      fill2DWithinLimits(h_efficiencyElectronPhi_vs_Eta_total_[threshold], recoEta, recoPhi);
      if (l1Et > threshold + probeToL1Offset_) {
        fill2DWithinLimits(h_efficiencyElectronPhi_vs_Eta_pass_[threshold], recoEta, recoPhi);
      }
    }
  }

  if (std::abs(recoEta) <= 1.479) {  // barrel
    // et
    fill2DWithinLimits(h_L1EGammaETvsElectronET_EB_, recoEt, l1Et);
    fill2DWithinLimits(h_L1EGammaETvsElectronET_EB_EE_, recoEt, l1Et);
    //resolution
    fillWithinLimits(h_resolutionElectronET_EB_, resolutionEt);
    fillWithinLimits(h_resolutionElectronET_EB_EE_, resolutionEt);
    // phi
    fill2DWithinLimits(h_L1EGammaPhivsElectronPhi_EB_, recoPhi, l1Phi);
    fill2DWithinLimits(h_L1EGammaPhivsElectronPhi_EB_EE_, recoPhi, l1Phi);
    // resolution
    fillWithinLimits(h_resolutionElectronPhi_EB_, resolutionPhi);
    fillWithinLimits(h_resolutionElectronPhi_EB_EE_, resolutionPhi);

    // turn-ons
    for (auto threshold : electronEfficiencyThresholds_) {
      fillWithinLimits(h_efficiencyElectronET_EB_total_[threshold], recoEt);
      fillWithinLimits(h_efficiencyElectronET_EB_EE_total_[threshold], recoEt);
      if (l1Et > threshold) {
        fillWithinLimits(h_efficiencyElectronET_EB_pass_[threshold], recoEt);
        fillWithinLimits(h_efficiencyElectronET_EB_EE_pass_[threshold], recoEt);
      }
    }
  } else {  // end-cap
    // et
    fill2DWithinLimits(h_L1EGammaETvsElectronET_EE_, recoEt, l1Et);
    fill2DWithinLimits(h_L1EGammaETvsElectronET_EB_EE_, recoEt, l1Et);
    //resolution
    fillWithinLimits(h_resolutionElectronET_EE_, resolutionEt);
    fillWithinLimits(h_resolutionElectronET_EB_EE_, resolutionEt);
    // phi
    fill2DWithinLimits(h_L1EGammaPhivsElectronPhi_EE_, recoPhi, l1Phi);
    fill2DWithinLimits(h_L1EGammaPhivsElectronPhi_EB_EE_, recoPhi, l1Phi);
    // resolution
    fillWithinLimits(h_resolutionElectronPhi_EE_, resolutionPhi);
    fillWithinLimits(h_resolutionElectronPhi_EB_EE_, resolutionPhi);

    // turn-ons
    for (auto threshold : electronEfficiencyThresholds_) {
      fillWithinLimits(h_efficiencyElectronET_EE_total_[threshold], recoEt);
      fillWithinLimits(h_efficiencyElectronET_EB_EE_total_[threshold], recoEt);
      if (l1Et > threshold) {
        fillWithinLimits(h_efficiencyElectronET_EE_pass_[threshold], recoEt);
        fillWithinLimits(h_efficiencyElectronET_EB_EE_pass_[threshold], recoEt);
      }
    }
  }
}

/**
 * From https://cds.cern.ch/record/2202966/files/DP2016_044.pdf slide 8
 * Filter on
 * HLT_Ele30WP60_Ele8_Mass55 (TODO)
 * HLT_Ele30WP60_SC4_Mass55 (TODO)
 * Seeded by L1SingleEG, unprescaled required
 *
 * Tag & probe selection
 * Electron required to be within ECAL fiducial volume (|η|<1.4442 ||
 * 1.566<|η|<2.5).
 * 60 < m(ee) < 120 GeV.
 * Opposite charge requirement.
 * Tag required to pass medium electron ID and ET > 30 GeV.
 * Probe required to pass loose electron ID.
 *
 * @param electrons
 * @return
 */
bool L1TEGammaOffline::findTagAndProbePair(edm::Handle<reco::GsfElectronCollection> const& electrons) {
  bool foundBoth(false);
  auto nElectrons = electrons->size();
  if (nElectrons < 2)
    return false;

  for (auto tagElectron : *electrons) {
    for (auto probeElectron : *electrons) {
      if (tagElectron.p4() == probeElectron.p4())
        continue;

      auto combined(tagElectron.p4() + probeElectron.p4());
      auto tagAbsEta = std::abs(tagElectron.eta());
      auto probeAbsEta = std::abs(probeElectron.eta());

      // EB-EE transition region
      bool isEBEEGap = tagElectron.isEBEEGap() || probeElectron.isEBEEGap();
      bool passesEta = !isEBEEGap && tagAbsEta < 2.5 && probeAbsEta < 2.5;
      bool passesCharge = tagElectron.charge() == -probeElectron.charge();

      // https://github.com/ikrav/cmssw/blob/egm_id_80X_v1/RecoEgamma/ElectronIdentification/plugins/cuts/GsfEleFull5x5SigmaIEtaIEtaCut.cc#L45
      bool tagPassesMediumID = passesMediumEleId(tagElectron) && tagElectron.et() > 30.;
      bool probePassesLooseID = passesLooseEleId(probeElectron);
      bool passesInvariantMass = combined.M() > 60 && combined.M() < 120;
      bool tagMatchesHLTObject = matchesAnHLTObject(tagElectron.eta(), tagElectron.phi());

      if (passesEta && passesInvariantMass && passesCharge && tagPassesMediumID && probePassesLooseID &&
          tagMatchesHLTObject) {
        foundBoth = true;
        tagElectron_ = tagElectron;
        probeElectron_ = probeElectron;
        // plot tag & probe invariant mass
        dqmoffline::l1t::fillWithinLimits(h_tagAndProbeMass_, combined.M());
        return foundBoth;
      }
    }
  }
  return foundBoth;
}

/**
 * Structure from
 * https://github.com/cms-sw/cmssw/blob/CMSSW_9_0_X/DQMOffline/EGamma/plugins/ElectronAnalyzer.cc
 * Values from
 * https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
 */
bool L1TEGammaOffline::passesLooseEleId(reco::GsfElectron const& electron) const {
  const float ecal_energy_inverse = 1.0 / electron.ecalEnergy();
  const float eSCoverP = electron.eSuperClusterOverP();
  const float eOverP = std::abs(1.0 - eSCoverP) * ecal_energy_inverse;

  if (electron.isEB() && eOverP > 0.241)
    return false;
  if (electron.isEE() && eOverP > 0.14)
    return false;
  if (electron.isEB() && std::abs(electron.deltaEtaSuperClusterTrackAtVtx()) > 0.00477)
    return false;
  if (electron.isEE() && std::abs(electron.deltaEtaSuperClusterTrackAtVtx()) > 0.00868)
    return false;
  if (electron.isEB() && std::abs(electron.deltaPhiSuperClusterTrackAtVtx()) > 0.222)
    return false;
  if (electron.isEE() && std::abs(electron.deltaPhiSuperClusterTrackAtVtx()) > 0.213)
    return false;
  if (electron.isEB() && electron.scSigmaIEtaIEta() > 0.011)
    return false;
  if (electron.isEE() && electron.scSigmaIEtaIEta() > 0.0314)
    return false;
  if (electron.isEB() && electron.hadronicOverEm() > 0.298)
    return false;
  if (electron.isEE() && electron.hadronicOverEm() > 0.101)
    return false;
  return true;
}

/*
 * Structure from
 * https://github.com/cms-sw/cmssw/blob/CMSSW_9_0_X/DQMOffline/EGamma/plugins/ElectronAnalyzer.cc
 * Values from
 * https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
 */
bool L1TEGammaOffline::passesMediumEleId(reco::GsfElectron const& electron) const {
  const float ecal_energy_inverse = 1.0 / electron.ecalEnergy();
  const float eSCoverP = electron.eSuperClusterOverP();
  const float eOverP = std::abs(1.0 - eSCoverP) * ecal_energy_inverse;

  if (electron.isEB() && eOverP < 0.134)
    return false;
  if (electron.isEE() && eOverP > 0.13)
    return false;
  if (electron.isEB() && std::abs(electron.deltaEtaSuperClusterTrackAtVtx()) > 0.00311)
    return false;
  if (electron.isEE() && std::abs(electron.deltaEtaSuperClusterTrackAtVtx()) > 0.00609)
    return false;
  if (electron.isEB() && std::abs(electron.deltaPhiSuperClusterTrackAtVtx()) > 0.103)
    return false;
  if (electron.isEE() && std::abs(electron.deltaPhiSuperClusterTrackAtVtx()) > 0.045)
    return false;
  if (electron.isEB() && electron.scSigmaIEtaIEta() > 0.00998)
    return false;
  if (electron.isEE() && electron.scSigmaIEtaIEta() > 0.0298)
    return false;
  if (electron.isEB() && electron.hadronicOverEm() > 0.253)
    return false;
  if (electron.isEE() && electron.hadronicOverEm() > 0.0878)
    return false;
  return true;
}

bool L1TEGammaOffline::matchesAnHLTObject(double eta, double phi) const {
  // get HLT objects of fired triggers
  using namespace dqmoffline::l1t;
  std::vector<bool> results = getTriggerResults(triggerIndices_, triggerResults_);
  std::vector<unsigned int> firedTriggers = getFiredTriggerIndices(triggerIndices_, results);
  std::vector<edm::InputTag> hltFilters = getHLTFilters(firedTriggers, hltConfig_, triggerProcess_);
  const trigger::TriggerObjectCollection hltObjects = getTriggerObjects(hltFilters, triggerEvent_);
  const trigger::TriggerObjectCollection matchedObjects =
      getMatchedTriggerObjects(eta, phi, maxDeltaRForHLTMatching_, hltObjects);

  return !matchedObjects.empty();
}

void L1TEGammaOffline::fillPhotons(edm::Event const& e, const unsigned int nVertex) {
  // TODO - just an example here
  edm::Handle<l1t::EGammaBxCollection> l1EGamma;
  e.getByToken(stage2CaloLayer2EGammaToken_, l1EGamma);

  edm::Handle<reco::PhotonCollection> photons;
  e.getByToken(thePhotonCollection_, photons);

  if (!photons.isValid()) {
    edm::LogWarning("L1TEGammaOffline") << "invalid collection: reco::Photons " << std::endl;
    return;
  }
  if (!l1EGamma.isValid()) {
    edm::LogWarning("L1TEGammaOffline") << "invalid collection: L1 EGamma " << std::endl;
    return;
  }

  if (photons->empty()) {
    LogDebug("L1TEGammaOffline") << "No photons found in event." << std::endl;
    return;
  }

  auto probePhoton = photons->at(0);

  double minDeltaR = 0.3;
  l1t::EGamma closestL1EGamma;
  bool foundMatch = false;

  int bunchCrossing = 0;
  for (auto egamma = l1EGamma->begin(bunchCrossing); egamma != l1EGamma->end(bunchCrossing); ++egamma) {
    double currentDeltaR = deltaR(egamma->eta(), egamma->phi(), probePhoton.eta(), probePhoton.phi());
    if (currentDeltaR > minDeltaR) {
      continue;
    } else {
      minDeltaR = currentDeltaR;
      closestL1EGamma = *egamma;
      foundMatch = true;
    }
  }

  if (!foundMatch) {
    LogDebug("L1TEGammaOffline") << "Could not find a matching L1 EGamma " << std::endl;
    return;
  }

  double recoEt = probePhoton.et();
  double recoEta = probePhoton.eta();
  double recoPhi = probePhoton.phi();

  double l1Et = closestL1EGamma.et();
  double l1Eta = closestL1EGamma.eta();
  double l1Phi = closestL1EGamma.phi();

  // if no reco value, relative resolution does not make sense -> sort to overflow
  double outOfBounds = 9999;
  double resolutionEt = recoEt > 0 ? (l1Et - recoEt) / recoEt : outOfBounds;
  double resolutionEta = l1Eta - recoEta;
  double resolutionPhi = reco::deltaPhi(l1Phi, recoPhi);

  using namespace dqmoffline::l1t;
  // eta
  fill2DWithinLimits(h_L1EGammaEtavsPhotonEta_, recoEta, l1Eta);
  fillWithinLimits(h_resolutionPhotonEta_, resolutionEta);

  if (std::abs(recoEta) <= 1.479) {  // barrel
    // et
    fill2DWithinLimits(h_L1EGammaETvsPhotonET_EB_, recoEt, l1Et);
    fill2DWithinLimits(h_L1EGammaETvsPhotonET_EB_EE_, recoEt, l1Et);
    //resolution
    fillWithinLimits(h_resolutionPhotonET_EB_, resolutionEt);
    fillWithinLimits(h_resolutionPhotonET_EB_EE_, resolutionEt);
    // phi
    fill2DWithinLimits(h_L1EGammaPhivsPhotonPhi_EB_, recoPhi, l1Phi);
    fill2DWithinLimits(h_L1EGammaPhivsPhotonPhi_EB_EE_, recoPhi, l1Phi);
    // resolution
    fillWithinLimits(h_resolutionPhotonPhi_EB_, resolutionPhi);
    fillWithinLimits(h_resolutionPhotonPhi_EB_EE_, resolutionPhi);

    // turn-ons
    for (auto threshold : photonEfficiencyThresholds_) {
      fillWithinLimits(h_efficiencyPhotonET_EB_total_[threshold], recoEt);
      fillWithinLimits(h_efficiencyPhotonET_EB_EE_total_[threshold], recoEt);
      if (l1Et > threshold) {
        fillWithinLimits(h_efficiencyPhotonET_EB_pass_[threshold], recoEt);
        fillWithinLimits(h_efficiencyPhotonET_EB_EE_pass_[threshold], recoEt);
      }
    }
  } else {  // end-cap
    // et
    fill2DWithinLimits(h_L1EGammaETvsPhotonET_EE_, recoEt, l1Et);
    fill2DWithinLimits(h_L1EGammaETvsPhotonET_EB_EE_, recoEt, l1Et);
    //resolution
    fillWithinLimits(h_resolutionPhotonET_EE_, resolutionEt);
    fillWithinLimits(h_resolutionPhotonET_EB_EE_, resolutionEt);
    // phi
    fill2DWithinLimits(h_L1EGammaPhivsPhotonPhi_EE_, recoPhi, l1Phi);
    fill2DWithinLimits(h_L1EGammaPhivsPhotonPhi_EB_EE_, recoPhi, l1Phi);
    // resolution
    fillWithinLimits(h_resolutionPhotonPhi_EE_, resolutionPhi);
    fillWithinLimits(h_resolutionPhotonPhi_EB_EE_, resolutionPhi);

    // turn-ons
    for (auto threshold : photonEfficiencyThresholds_) {
      fillWithinLimits(h_efficiencyPhotonET_EE_total_[threshold], recoEt);
      fillWithinLimits(h_efficiencyPhotonET_EB_EE_total_[threshold], recoEt);
      if (l1Et > threshold) {
        fillWithinLimits(h_efficiencyPhotonET_EE_pass_[threshold], recoEt);
        fillWithinLimits(h_efficiencyPhotonET_EB_EE_pass_[threshold], recoEt);
      }
    }
  }
}

//
// -------------------------------------- endRun --------------------------------------------
//
void L1TEGammaOffline::dqmEndRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  normalise2DHistogramsToBinArea();
}

//
// -------------------------------------- book histograms --------------------------------------------
//
void L1TEGammaOffline::bookElectronHistos(DQMStore::IBooker& ibooker) {
  ibooker.cd();
  ibooker.setCurrentFolder(histFolder_);
  // since there are endRun manipulations.
  ibooker.setScope(MonitorElementData::Scope::RUN);

  dqmoffline::l1t::HistDefinition nVertexDef = histDefinitions_[PlotConfig::nVertex];
  h_nVertex_ = ibooker.book1D(nVertexDef.name, nVertexDef.title, nVertexDef.nbinsX, nVertexDef.xmin, nVertexDef.xmax);
  h_tagAndProbeMass_ = ibooker.book1D("tagAndProbeMass", "Invariant mass of tag & probe pair", 100, 40, 140);
  // electron reco vs L1
  dqmoffline::l1t::HistDefinition templateETvsET = histDefinitions_[PlotConfig::ETvsET];
  h_L1EGammaETvsElectronET_EB_ =
      ibooker.book2D("L1EGammaETvsElectronET_EB",
                     "L1 EGamma E_{T} vs GSF Electron E_{T} (EB); GSF Electron E_{T} (GeV); L1 EGamma E_{T} (GeV)",
                     templateETvsET.nbinsX,
                     &templateETvsET.binsX[0],
                     templateETvsET.nbinsY,
                     &templateETvsET.binsY[0]);
  h_L1EGammaETvsElectronET_EE_ =
      ibooker.book2D("L1EGammaETvsElectronET_EE",
                     "L1 EGamma E_{T} vs GSF Electron E_{T} (EE); GSF Electron E_{T} (GeV); L1 EGamma E_{T} (GeV)",
                     templateETvsET.nbinsX,
                     &templateETvsET.binsX[0],
                     templateETvsET.nbinsY,
                     &templateETvsET.binsY[0]);
  h_L1EGammaETvsElectronET_EB_EE_ =
      ibooker.book2D("L1EGammaETvsElectronET_EB_EE",
                     "L1 EGamma E_{T} vs GSF Electron E_{T} (EB+EE); GSF Electron E_{T} (GeV); L1 EGamma E_{T} (GeV)",
                     templateETvsET.nbinsX,
                     &templateETvsET.binsX[0],
                     templateETvsET.nbinsY,
                     &templateETvsET.binsY[0]);

  dqmoffline::l1t::HistDefinition templatePHIvsPHI = histDefinitions_[PlotConfig::PHIvsPHI];
  h_L1EGammaPhivsElectronPhi_EB_ = ibooker.book2D(
      "L1EGammaPhivsElectronPhi_EB",
      "#phi_{electron}^{L1} vs #phi_{electron}^{offline} (EB); #phi_{electron}^{offline}; #phi_{electron}^{L1}",
      templatePHIvsPHI.nbinsX,
      templatePHIvsPHI.xmin,
      templatePHIvsPHI.xmax,
      templatePHIvsPHI.nbinsY,
      templatePHIvsPHI.ymin,
      templatePHIvsPHI.ymax);
  h_L1EGammaPhivsElectronPhi_EE_ = ibooker.book2D(
      "L1EGammaPhivsElectronPhi_EE",
      "#phi_{electron}^{L1} vs #phi_{electron}^{offline} (EE); #phi_{electron}^{offline}; #phi_{electron}^{L1}",
      templatePHIvsPHI.nbinsX,
      templatePHIvsPHI.xmin,
      templatePHIvsPHI.xmax,
      templatePHIvsPHI.nbinsY,
      templatePHIvsPHI.ymin,
      templatePHIvsPHI.ymax);
  h_L1EGammaPhivsElectronPhi_EB_EE_ = ibooker.book2D(
      "L1EGammaPhivsElectronPhi_EB_EE",
      "#phi_{electron}^{L1} vs #phi_{electron}^{offline} (EB+EE); #phi_{electron}^{offline}; #phi_{electron}^{L1}",
      templatePHIvsPHI.nbinsX,
      templatePHIvsPHI.xmin,
      templatePHIvsPHI.xmax,
      templatePHIvsPHI.nbinsY,
      templatePHIvsPHI.ymin,
      templatePHIvsPHI.ymax);

  h_L1EGammaEtavsElectronEta_ = ibooker.book2D("L1EGammaEtavsElectronEta",
                                               "L1 EGamma #eta vs GSF Electron #eta; GSF Electron #eta; L1 EGamma #eta",
                                               100,
                                               -3,
                                               3,
                                               100,
                                               -3,
                                               3);

  // electron resolutions
  h_resolutionElectronET_EB_ =
      ibooker.book1D("resolutionElectronET_EB",
                     "electron ET resolution (EB); (L1 EGamma E_{T} - GSF Electron E_{T})/GSF Electron E_{T}; events",
                     50,
                     -1,
                     1.5);
  h_resolutionElectronET_EE_ =
      ibooker.book1D("resolutionElectronET_EE",
                     "electron ET resolution (EE); (L1 EGamma E_{T} - GSF Electron E_{T})/GSF Electron E_{T}; events",
                     50,
                     -1,
                     1.5);
  h_resolutionElectronET_EB_EE_ = ibooker.book1D(
      "resolutionElectronET_EB_EE",
      "electron ET resolution (EB+EE); (L1 EGamma E_{T} - GSF Electron E_{T})/GSF Electron E_{T}; events",
      50,
      -1,
      1.5);

  h_resolutionElectronPhi_EB_ =
      ibooker.book1D("resolutionElectronPhi_EB",
                     "#phi_{electron} resolution (EB); #phi_{electron}^{L1} - #phi_{electron}^{offline}; events",
                     120,
                     -0.3,
                     0.3);
  h_resolutionElectronPhi_EE_ =
      ibooker.book1D("resolutionElectronPhi_EE",
                     "electron #phi resolution (EE); #phi_{electron}^{L1} - #phi_{electron}^{offline}; events",
                     120,
                     -0.3,
                     0.3);
  h_resolutionElectronPhi_EB_EE_ =
      ibooker.book1D("resolutionElectronPhi_EB_EE",
                     "electron #phi resolution (EB+EE); #phi_{electron}^{L1} - #phi_{electron}^{offline}; events",
                     120,
                     -0.3,
                     0.3);

  h_resolutionElectronEta_ =
      ibooker.book1D("resolutionElectronEta",
                     "electron #eta resolution  (EB); L1 EGamma #eta - GSF Electron #eta; events",
                     120,
                     -0.3,
                     0.3);

  // electron turn-ons
  ibooker.setCurrentFolder(efficiencyFolder_);
  std::vector<float> electronBins(electronEfficiencyBins_.begin(), electronEfficiencyBins_.end());
  int nBins = electronBins.size() - 1;
  float* electronBinArray = &(electronBins[0]);

  for (auto threshold : electronEfficiencyThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyElectronET_EB_pass_[threshold] =
        ibooker.book1D("efficiencyElectronET_EB_threshold_" + str_threshold + "_Num",
                       "electron efficiency (EB) (numerator); GSF Electron E_{T} (GeV); events",
                       nBins,
                       electronBinArray);
    h_efficiencyElectronET_EE_pass_[threshold] =
        ibooker.book1D("efficiencyElectronET_EE_threshold_" + str_threshold + "_Num",
                       "electron efficiency (EE) (numerator); GSF Electron E_{T} (GeV); events",
                       nBins,
                       electronBinArray);
    h_efficiencyElectronET_EB_EE_pass_[threshold] =
        ibooker.book1D("efficiencyElectronET_EB_EE_threshold_" + str_threshold + "_Num",
                       "electron efficiency (EB+EE) (numerator); GSF Electron E_{T} (GeV); events",
                       nBins,
                       electronBinArray);
    h_efficiencyElectronPhi_vs_Eta_pass_[threshold] =
        ibooker.book2D("efficiencyElectronPhi_vs_Eta_threshold_" + str_threshold + "_Num",
                       "electron efficiency (numerator); GSF Electron #eta; GSF Electron #phi",
                       50,
                       -2.5,
                       2.5,
                       32,
                       -3.2,
                       3.2);

    h_efficiencyElectronET_EB_total_[threshold] =
        ibooker.book1D("efficiencyElectronET_EB_threshold_" + str_threshold + "_Den",
                       "electron efficiency (EB) (denominator); GSF Electron E_{T} (GeV); events",
                       nBins,
                       electronBinArray);
    h_efficiencyElectronET_EE_total_[threshold] =
        ibooker.book1D("efficiencyElectronET_EE_threshold_" + str_threshold + "_Den",
                       "electron efficiency (EE) (denominator); GSF Electron E_{T} (GeV); events",
                       nBins,
                       electronBinArray);
    h_efficiencyElectronET_EB_EE_total_[threshold] =
        ibooker.book1D("efficiencyElectronET_EB_EE_threshold_" + str_threshold + "_Den",
                       "electron efficiency (EB+EE) (denominator); GSF Electron E_{T} (GeV); events",
                       nBins,
                       electronBinArray);
    h_efficiencyElectronPhi_vs_Eta_total_[threshold] =
        ibooker.book2D("efficiencyElectronPhi_vs_Eta_threshold_" + str_threshold + "_Den",
                       "electron efficiency (denominator); GSF Electron #eta; GSF Electron #phi",
                       50,
                       -2.5,
                       2.5,
                       32,
                       -3.2,
                       3.2);
  }

  for (auto threshold : deepInspectionElectronThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyElectronEta_pass_[threshold] =
        ibooker.book1D("efficiencyElectronEta_threshold_" + str_threshold + "_Num",
                       "electron efficiency (numerator); GSF Electron #eta; events",
                       50,
                       -2.5,
                       2.5);
    h_efficiencyElectronPhi_pass_[threshold] =
        ibooker.book1D("efficiencyElectronPhi_threshold_" + str_threshold + "_Num",
                       "electron efficiency (numerator); GSF Electron #phi; events",
                       32,
                       -3.2,
                       3.2);
    h_efficiencyElectronNVertex_pass_[threshold] =
        ibooker.book1D("efficiencyElectronNVertex_threshold_" + str_threshold + "_Num",
                       "electron efficiency (numerator); Nvtx; events",
                       30,
                       0,
                       60);

    h_efficiencyElectronEta_total_[threshold] =
        ibooker.book1D("efficiencyElectronEta_threshold_" + str_threshold + "_Den",
                       "electron efficiency (denominator); GSF Electron #eta; events",
                       50,
                       -2.5,
                       2.5);
    h_efficiencyElectronPhi_total_[threshold] =
        ibooker.book1D("efficiencyElectronPhi_threshold_" + str_threshold + "_Den",
                       "electron efficiency (denominator); GSF Electron #phi; events",
                       32,
                       -3.2,
                       3.2);
    h_efficiencyElectronNVertex_total_[threshold] =
        ibooker.book1D("efficiencyElectronNVertex_threshold_" + str_threshold + "_Den",
                       "electron efficiency (denominator); Nvtx; events",
                       30,
                       0,
                       60);
  }

  ibooker.cd();
}

void L1TEGammaOffline::bookPhotonHistos(DQMStore::IBooker& ibooker) {
  ibooker.cd();
  ibooker.setCurrentFolder(histFolder_);

  dqmoffline::l1t::HistDefinition templateETvsET = histDefinitions_[PlotConfig::ETvsET];
  h_L1EGammaETvsPhotonET_EB_ =
      ibooker.book2D("L1EGammaETvsPhotonET_EB",
                     "L1 EGamma E_{T} vs  Photon E_{T} (EB);  Photon E_{T} (GeV); L1 EGamma E_{T} (GeV)",
                     templateETvsET.nbinsX,
                     &templateETvsET.binsX[0],
                     templateETvsET.nbinsY,
                     &templateETvsET.binsY[0]);
  h_L1EGammaETvsPhotonET_EE_ =
      ibooker.book2D("L1EGammaETvsPhotonET_EE",
                     "L1 EGamma E_{T} vs  Photon E_{T} (EE);  Photon E_{T} (GeV); L1 EGamma E_{T} (GeV)",
                     templateETvsET.nbinsX,
                     &templateETvsET.binsX[0],
                     templateETvsET.nbinsY,
                     &templateETvsET.binsY[0]);
  h_L1EGammaETvsPhotonET_EB_EE_ =
      ibooker.book2D("L1EGammaETvsPhotonET_EB_EE",
                     "L1 EGamma E_{T} vs  Photon E_{T} (EB+EE);  Photon E_{T} (GeV); L1 EGamma E_{T} (GeV)",
                     templateETvsET.nbinsX,
                     &templateETvsET.binsX[0],
                     templateETvsET.nbinsY,
                     &templateETvsET.binsY[0]);

  dqmoffline::l1t::HistDefinition templatePHIvsPHI = histDefinitions_[PlotConfig::PHIvsPHI];
  h_L1EGammaPhivsPhotonPhi_EB_ =
      ibooker.book2D("L1EGammaPhivsPhotonPhi_EB",
                     "#phi_{photon}^{L1} vs #phi_{photon}^{offline} (EB); #phi_{photon}^{offline}; #phi_{photon}^{L1}",
                     templatePHIvsPHI.nbinsX,
                     templatePHIvsPHI.xmin,
                     templatePHIvsPHI.xmax,
                     templatePHIvsPHI.nbinsY,
                     templatePHIvsPHI.ymin,
                     templatePHIvsPHI.ymax);
  h_L1EGammaPhivsPhotonPhi_EE_ =
      ibooker.book2D("L1EGammaPhivsPhotonPhi_EE",
                     "#phi_{photon}^{L1} vs #phi_{photon}^{offline} (EE); #phi_{photon}^{offline}; #phi_{photon}^{L1}",
                     templatePHIvsPHI.nbinsX,
                     templatePHIvsPHI.xmin,
                     templatePHIvsPHI.xmax,
                     templatePHIvsPHI.nbinsY,
                     templatePHIvsPHI.ymin,
                     templatePHIvsPHI.ymax);
  h_L1EGammaPhivsPhotonPhi_EB_EE_ = ibooker.book2D(
      "L1EGammaPhivsPhotonPhi_EB_EE",
      "#phi_{photon}^{L1} vs #phi_{photon}^{offline} (EB+EE); #phi_{photon}^{offline}; #phi_{photon}^{L1}",
      templatePHIvsPHI.nbinsX,
      templatePHIvsPHI.xmin,
      templatePHIvsPHI.xmax,
      templatePHIvsPHI.nbinsY,
      templatePHIvsPHI.ymin,
      templatePHIvsPHI.ymax);

  h_L1EGammaEtavsPhotonEta_ = ibooker.book2D(
      "L1EGammaEtavsPhotonEta", "L1 EGamma #eta vs  Photon #eta;  Photon #eta; L1 EGamma #eta", 100, -3, 3, 100, -3, 3);

  // photon resolutions
  h_resolutionPhotonET_EB_ =
      ibooker.book1D("resolutionPhotonET_EB",
                     "photon ET resolution (EB); (L1 EGamma E_{T} -  Photon E_{T})/ Photon E_{T}; events",
                     50,
                     -1,
                     1.5);
  h_resolutionPhotonET_EE_ =
      ibooker.book1D("resolutionPhotonET_EE",
                     "photon ET resolution (EE); (L1 EGamma E_{T} -  Photon E_{T})/ Photon E_{T}; events",
                     50,
                     -1,
                     1.5);
  h_resolutionPhotonET_EB_EE_ =
      ibooker.book1D("resolutionPhotonET_EB_EE",
                     "photon ET resolution (EB+EE); (L1 EGamma E_{T} -  Photon E_{T})/ Photon E_{T}; events",
                     50,
                     -1,
                     1.5);

  h_resolutionPhotonPhi_EB_ =
      ibooker.book1D("resolutionPhotonPhi_EB",
                     "#phi_{photon} resolution (EB); #phi_{photon}^{L1} - #phi_{photon}^{offline}; events",
                     120,
                     -0.3,
                     0.3);
  h_resolutionPhotonPhi_EE_ =
      ibooker.book1D("resolutionPhotonPhi_EE",
                     "photon #phi resolution (EE); #phi_{photon}^{L1} - #phi_{photon}^{offline}; events",
                     120,
                     -0.3,
                     0.3);
  h_resolutionPhotonPhi_EB_EE_ =
      ibooker.book1D("resolutionPhotonPhi_EB_EE",
                     "photon #phi resolution (EB+EE); #phi_{photon}^{L1} - #phi_{photon}^{offline}; events",
                     120,
                     -0.3,
                     0.3);

  h_resolutionPhotonEta_ = ibooker.book1D(
      "resolutionPhotonEta", "photon #eta resolution  (EB); L1 EGamma #eta -  Photon #eta; events", 120, -0.3, 0.3);

  // photon turn-ons
  ibooker.setCurrentFolder(efficiencyFolder_);
  std::vector<float> photonBins(photonEfficiencyBins_.begin(), photonEfficiencyBins_.end());
  int nBins = photonBins.size() - 1;
  float* photonBinArray = &(photonBins[0]);

  for (auto threshold : photonEfficiencyThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyPhotonET_EB_pass_[threshold] =
        ibooker.book1D("efficiencyPhotonET_EB_threshold_" + str_threshold + "_Num",
                       "photon efficiency (EB) (numerator);  Photon E_{T} (GeV); events",
                       nBins,
                       photonBinArray);
    h_efficiencyPhotonET_EE_pass_[threshold] =
        ibooker.book1D("efficiencyPhotonET_EE_threshold_" + str_threshold + "_Num",
                       "photon efficiency (EE) (numerator);  Photon E_{T} (GeV); events",
                       nBins,
                       photonBinArray);
    h_efficiencyPhotonET_EB_EE_pass_[threshold] =
        ibooker.book1D("efficiencyPhotonET_EB_EE_threshold_" + str_threshold + "_Num",
                       "photon efficiency (EB+EE) (numerator);  Photon E_{T} (GeV); events",
                       nBins,
                       photonBinArray);

    h_efficiencyPhotonET_EB_total_[threshold] =
        ibooker.book1D("efficiencyPhotonET_EB_threshold_" + str_threshold + "_Den",
                       "photon efficiency (EB) (denominator);  Photon E_{T} (GeV); events",
                       nBins,
                       photonBinArray);
    h_efficiencyPhotonET_EE_total_[threshold] =
        ibooker.book1D("efficiencyPhotonET_EE_threshold_" + str_threshold + "_Den",
                       "photon efficiency (EE) (denominator);  Photon E_{T} (GeV); events",
                       nBins,
                       photonBinArray);
    h_efficiencyPhotonET_EB_EE_total_[threshold] =
        ibooker.book1D("efficiencyPhotonET_EB_EE_threshold_" + str_threshold + "_Den",
                       "photon efficiency (EB+EE) (denominator);  Photon E_{T} (GeV); events",
                       nBins,
                       photonBinArray);
  }

  ibooker.cd();
}

void L1TEGammaOffline::normalise2DHistogramsToBinArea() {
  std::vector<MonitorElement*> monElementstoNormalize = {h_L1EGammaETvsElectronET_EB_,
                                                         h_L1EGammaETvsElectronET_EE_,
                                                         h_L1EGammaETvsElectronET_EB_EE_,
                                                         h_L1EGammaPhivsElectronPhi_EB_,
                                                         h_L1EGammaPhivsElectronPhi_EE_,
                                                         h_L1EGammaPhivsElectronPhi_EB_EE_,
                                                         h_L1EGammaEtavsElectronEta_,
                                                         h_L1EGammaETvsPhotonET_EB_,
                                                         h_L1EGammaETvsPhotonET_EE_,
                                                         h_L1EGammaETvsPhotonET_EB_EE_,
                                                         h_L1EGammaPhivsPhotonPhi_EB_,
                                                         h_L1EGammaPhivsPhotonPhi_EE_,
                                                         h_L1EGammaPhivsPhotonPhi_EB_EE_,
                                                         h_L1EGammaEtavsPhotonEta_};

  for (auto mon : monElementstoNormalize) {
    if (mon != nullptr) {
      auto h = mon->getTH2F();
      if (h != nullptr) {
        h->Scale(1, "width");
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TEGammaOffline);
