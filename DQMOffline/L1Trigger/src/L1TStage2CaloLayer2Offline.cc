#include "DQMOffline/L1Trigger/interface/L1TStage2CaloLayer2Offline.h"
#include "DQMOffline/L1Trigger/interface/L1TFillWithinLimits.h"
#include "DQMOffline/L1Trigger/interface/L1TCommon.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
// Geometry
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "TLorentzVector.h"

const std::map<std::string, unsigned int> L1TStage2CaloLayer2Offline::PlotConfigNames = {
    {"nVertex", PlotConfig::nVertex}, {"ETvsET", PlotConfig::ETvsET}, {"PHIvsPHI", PlotConfig::PHIvsPHI}};

//
// -------------------------------------- Constructor --------------------------------------------
//
L1TStage2CaloLayer2Offline::L1TStage2CaloLayer2Offline(const edm::ParameterSet& ps)
    : thePFJetCollection_(consumes<reco::PFJetCollection>(ps.getParameter<edm::InputTag>("pfJetCollection"))),
      thecaloMETCollection_(consumes<reco::CaloMETCollection>(ps.getParameter<edm::InputTag>("caloMETCollection"))),
      thecaloETMHFCollection_(consumes<reco::CaloMETCollection>(ps.getParameter<edm::InputTag>("caloETMHFCollection"))),
      thePFMETNoMuCollection_(consumes<reco::PFMETCollection>(ps.getParameter<edm::InputTag>("pfMETNoMuCollection"))),
      thePVCollection_(consumes<reco::VertexCollection>(ps.getParameter<edm::InputTag>("PVCollection"))),
      theBSCollection_(consumes<reco::BeamSpot>(ps.getParameter<edm::InputTag>("beamSpotCollection"))),
      triggerInputTag_(consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("triggerInputTag"))),
      triggerResultsInputTag_(consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("triggerResults"))),
      triggerProcess_(ps.getParameter<std::string>("triggerProcess")),
      triggerNames_(ps.getParameter<std::vector<std::string>>("triggerNames")),
      histFolderEtSum_(ps.getParameter<std::string>("histFolderEtSum")),
      histFolderJet_(ps.getParameter<std::string>("histFolderJet")),
      efficiencyFolderEtSum_(histFolderEtSum_ + "/efficiency_raw"),
      efficiencyFolderJet_(histFolderJet_ + "/efficiency_raw"),
      stage2CaloLayer2JetToken_(
          consumes<l1t::JetBxCollection>(ps.getParameter<edm::InputTag>("stage2CaloLayer2JetSource"))),
      stage2CaloLayer2EtSumToken_(
          consumes<l1t::EtSumBxCollection>(ps.getParameter<edm::InputTag>("stage2CaloLayer2EtSumSource"))),
      jetEfficiencyThresholds_(ps.getParameter<std::vector<double>>("jetEfficiencyThresholds")),
      metEfficiencyThresholds_(ps.getParameter<std::vector<double>>("metEfficiencyThresholds")),
      mhtEfficiencyThresholds_(ps.getParameter<std::vector<double>>("mhtEfficiencyThresholds")),
      ettEfficiencyThresholds_(ps.getParameter<std::vector<double>>("ettEfficiencyThresholds")),
      httEfficiencyThresholds_(ps.getParameter<std::vector<double>>("httEfficiencyThresholds")),
      jetEfficiencyBins_(ps.getParameter<std::vector<double>>("jetEfficiencyBins")),
      metEfficiencyBins_(ps.getParameter<std::vector<double>>("metEfficiencyBins")),
      mhtEfficiencyBins_(ps.getParameter<std::vector<double>>("mhtEfficiencyBins")),
      ettEfficiencyBins_(ps.getParameter<std::vector<double>>("ettEfficiencyBins")),
      httEfficiencyBins_(ps.getParameter<std::vector<double>>("httEfficiencyBins")),
      recoHTTMaxEta_(ps.getParameter<double>("recoHTTMaxEta")),
      recoMHTMaxEta_(ps.getParameter<double>("recoMHTMaxEta")),
      hltConfig_(),
      triggerIndices_(),
      triggerResults_(),
      triggerEvent_(),
      histDefinitions_(dqmoffline::l1t::readHistDefinitions(ps.getParameterSet("histDefinitions"), PlotConfigNames)),
      h_nVertex_(),
      h_controlPlots_(),
      h_L1METvsCaloMET_(),
      h_L1ETMHFvsCaloETMHF_(),
      h_L1METvsPFMetNoMu_(),
      h_L1MHTvsRecoMHT_(),
      h_L1METTvsCaloETT_(),
      h_L1HTTvsRecoHTT_(),
      h_L1METPhivsCaloMETPhi_(),
      h_L1ETMHFPhivsCaloETMHFPhi_(),
      h_L1METPhivsPFMetNoMuPhi_(),
      h_L1MHTPhivsRecoMHTPhi_(),
      h_resolutionMET_(),
      h_resolutionETMHF_(),
      h_resolutionPFMetNoMu_(),
      h_resolutionMHT_(),
      h_resolutionETT_(),
      h_resolutionHTT_(),
      h_resolutionMETPhi_(),
      h_resolutionETMHFPhi_(),
      h_resolutionPFMetNoMuPhi_(),
      h_resolutionMHTPhi_(),
      h_efficiencyMET_pass_(),
      h_efficiencyETMHF_pass_(),
      h_efficiencyPFMetNoMu_pass_(),
      h_efficiencyMHT_pass_(),
      h_efficiencyETT_pass_(),
      h_efficiencyHTT_pass_(),
      h_efficiencyMET_total_(),
      h_efficiencyETMHF_total_(),
      h_efficiencyPFMetNoMu_total_(),
      h_efficiencyMHT_total_(),
      h_efficiencyETT_total_(),
      h_efficiencyHTT_total_(),
      h_L1JetETvsPFJetET_HB_(),
      h_L1JetETvsPFJetET_HE_(),
      h_L1JetETvsPFJetET_HF_(),
      h_L1JetETvsPFJetET_HB_HE_(),
      h_L1JetPhivsPFJetPhi_HB_(),
      h_L1JetPhivsPFJetPhi_HE_(),
      h_L1JetPhivsPFJetPhi_HF_(),
      h_L1JetPhivsPFJetPhi_HB_HE_(),
      h_L1JetEtavsPFJetEta_(),
      h_resolutionJetET_HB_(),
      h_resolutionJetET_HE_(),
      h_resolutionJetET_HF_(),
      h_resolutionJetET_HB_HE_(),
      h_resolutionJetPhi_HB_(),
      h_resolutionJetPhi_HE_(),
      h_resolutionJetPhi_HF_(),
      h_resolutionJetPhi_HB_HE_(),
      h_resolutionJetEta_(),
      h_efficiencyJetEt_HB_pass_(),
      h_efficiencyJetEt_HE_pass_(),
      h_efficiencyJetEt_HF_pass_(),
      h_efficiencyJetEt_HB_HE_pass_(),
      h_efficiencyJetEt_HB_total_(),
      h_efficiencyJetEt_HE_total_(),
      h_efficiencyJetEt_HF_total_(),
      h_efficiencyJetEt_HB_HE_total_() {
  edm::LogInfo("L1TStage2CaloLayer2Offline") << "Constructor "
                                             << "L1TStage2CaloLayer2Offline::L1TStage2CaloLayer2Offline " << std::endl;
}

//
// -- Destructor
//
L1TStage2CaloLayer2Offline::~L1TStage2CaloLayer2Offline() {
  edm::LogInfo("L1TStage2CaloLayer2Offline")
      << "Destructor L1TStage2CaloLayer2Offline::~L1TStage2CaloLayer2Offline " << std::endl;
}

//
// -------------------------------------- beginRun --------------------------------------------
//
void L1TStage2CaloLayer2Offline::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  edm::LogInfo("L1TStage2CaloLayer2Offline") << "L1TStage2CaloLayer2Offline::beginRun" << std::endl;
  bool changed(true);
  if (!hltConfig_.init(iRun, iSetup, triggerProcess_, changed)) {
    edm::LogError("L1TStage2CaloLayer2Offline")
        << " HLT config extraction failure with process name " << triggerProcess_ << std::endl;
    triggerNames_.clear();
  } else {
    triggerIndices_ = dqmoffline::l1t::getTriggerIndices(triggerNames_, hltConfig_.triggerNames());
  }
}
//
// -------------------------------------- bookHistos --------------------------------------------
//
void L1TStage2CaloLayer2Offline::bookHistograms(DQMStore::IBooker& ibooker_, edm::Run const&, edm::EventSetup const&) {
  edm::LogInfo("L1TStage2CaloLayer2Offline") << "L1TStage2CaloLayer2Offline::bookHistograms" << std::endl;

  //book at beginRun
  bookHistos(ibooker_);
}
//
// -------------------------------------- Analyze --------------------------------------------
//
void L1TStage2CaloLayer2Offline::analyze(edm::Event const& e, edm::EventSetup const& eSetup) {
  edm::LogInfo("L1TStage2CaloLayer2Offline") << "L1TStage2CaloLayer2Offline::analyze" << std::endl;

  edm::Handle<edm::TriggerResults> triggerResultHandle;
  e.getByToken(triggerResultsInputTag_, triggerResultHandle);
  if (!triggerResultHandle.isValid()) {
    edm::LogWarning("L1TStage2CaloLayer2Offline") << "invalid edm::TriggerResults handle" << std::endl;
    return;
  }
  triggerResults_ = *triggerResultHandle;

  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  e.getByToken(triggerInputTag_, triggerEventHandle);
  if (!triggerEventHandle.isValid()) {
    edm::LogWarning("L1TStage2CaloLayer2Offline") << "invalid trigger::TriggerEvent handle" << std::endl;
    return;
  }
  triggerEvent_ = *triggerEventHandle;

  edm::Handle<reco::VertexCollection> vertexHandle;
  e.getByToken(thePVCollection_, vertexHandle);
  if (!vertexHandle.isValid()) {
    edm::LogWarning("L1TStage2CaloLayer2Offline") << "invalid collection: vertex " << std::endl;
    return;
  }

  unsigned int nVertex = vertexHandle->size();
  dqmoffline::l1t::fillWithinLimits(h_nVertex_, nVertex);

  // L1T
  if (!dqmoffline::l1t::passesAnyTriggerFromList(triggerIndices_, triggerResults_)) {
    return;
  }
  fillEnergySums(e, nVertex);
  fillJets(e, nVertex);
}

void L1TStage2CaloLayer2Offline::fillEnergySums(edm::Event const& e, const unsigned int nVertex) {
  edm::Handle<l1t::EtSumBxCollection> l1EtSums;
  e.getByToken(stage2CaloLayer2EtSumToken_, l1EtSums);

  edm::Handle<reco::PFJetCollection> pfJets;
  e.getByToken(thePFJetCollection_, pfJets);

  edm::Handle<reco::CaloMETCollection> caloMETs;
  e.getByToken(thecaloMETCollection_, caloMETs);

  edm::Handle<reco::CaloMETCollection> caloETMHFs;
  e.getByToken(thecaloETMHFCollection_, caloETMHFs);

  edm::Handle<reco::PFMETCollection> pfMETNoMus;
  e.getByToken(thePFMETNoMuCollection_, pfMETNoMus);

  if (!pfJets.isValid()) {
    edm::LogWarning("L1TStage2CaloLayer2Offline") << "invalid collection: PF jets " << std::endl;
    return;
  }
  if (!caloMETs.isValid()) {
    edm::LogWarning("L1TStage2CaloLayer2Offline") << "invalid collection: Offline E_{T}^{miss} " << std::endl;
    return;
  }
  if (!caloETMHFs.isValid()) {
    edm::LogWarning("L1TStage2CaloLayer2Offline") << "invalid collection: Offline E_{T}^{miss} (HF) " << std::endl;
    return;
  }
  if (!pfMETNoMus.isValid()) {
    edm::LogWarning("L1TStage2CaloLayer2Offline") << "invalid collection: Offline PF E_{T}^{miss} No Mu" << std::endl;
    return;
  }
  if (!l1EtSums.isValid()) {
    edm::LogWarning("L1TStage2CaloLayer2Offline") << "invalid collection: L1 ET sums " << std::endl;
    return;
  }

  int bunchCrossing = 0;

  double l1MET(0);
  double l1METPhi(0);
  double l1ETMHF(0);
  double l1ETMHFPhi(0);
  double l1MHT(0);
  double l1MHTPhi(0);
  double l1ETT(0);
  double l1HTT(0);

  for (auto etSum = l1EtSums->begin(bunchCrossing); etSum != l1EtSums->end(bunchCrossing); ++etSum) {
    double et = etSum->et();
    double phi = etSum->phi();

    switch (etSum->getType()) {
      case l1t::EtSum::EtSumType::kMissingEt:
        l1MET = et;
        l1METPhi = phi;
        break;
      case l1t::EtSum::EtSumType::kMissingEtHF:
        l1ETMHF = et;
        l1ETMHFPhi = phi;
        break;
      case l1t::EtSum::EtSumType::kTotalEt:
        l1ETT = et;
        break;
      case l1t::EtSum::EtSumType::kMissingHt:
        l1MHT = et;
        l1MHTPhi = phi;
        break;
      case l1t::EtSum::EtSumType::kTotalHt:
        l1HTT = et;
      default:
        break;
    }
  }

  double recoMET(caloMETs->front().et());
  double recoMETPhi(caloMETs->front().phi());
  double recoETMHF(caloETMHFs->front().et());
  double recoETMHFPhi(caloETMHFs->front().phi());
  double recoPFMetNoMu(pfMETNoMus->front().et());
  double recoPFMetNoMuPhi(pfMETNoMus->front().phi());
  double recoMHT(0);
  double recoMHTPhi(0);
  double recoETT(caloMETs->front().sumEt());
  double recoHTT(0);

  TVector2 mht(0., 0.);

  for (auto jet = pfJets->begin(); jet != pfJets->end(); ++jet) {
    double et = jet->et();
    if (et < 30) {
      continue;
    }
    TVector2 jetVec(et * cos(jet->phi()), et * sin(jet->phi()));
    if (std::abs(jet->eta()) < recoHTTMaxEta_) {
      recoHTT += et;
    }
    if (std::abs(jet->eta()) < recoMHTMaxEta_) {
      mht -= jetVec;
    }
  }
  recoMHT = mht.Mod();
  // phi in cms is defined between -pi and pi
  recoMHTPhi = TVector2::Phi_mpi_pi(mht.Phi());

  // if no reco value, relative resolution does not make sense -> sort to overflow
  double outOfBounds = 9999;

  double resolutionMET = recoMET > 0 ? (l1MET - recoMET) / recoMET : outOfBounds;
  double resolutionMETPhi = reco::deltaPhi(l1METPhi, recoMETPhi);

  double resolutionETMHF = recoETMHF > 0 ? (l1ETMHF - recoETMHF) / recoETMHF : outOfBounds;
  double resolutionETMHFPhi = reco::deltaPhi(l1ETMHFPhi, recoETMHFPhi);

  double resolutionPFMetNoMu = recoETMHF > 0 ? (l1MET - recoPFMetNoMu) / recoPFMetNoMu : outOfBounds;
  double resolutionPFMetNoMuPhi = reco::deltaPhi(l1METPhi, recoPFMetNoMuPhi);

  double resolutionMHT = recoMHT > 0 ? (l1MHT - recoMHT) / recoMHT : outOfBounds;
  double resolutionMHTPhi = reco::deltaPhi(l1MHTPhi, recoMHTPhi);

  double resolutionETT = recoETT > 0 ? (l1ETT - recoETT) / recoETT : outOfBounds;
  double resolutionHTT = recoHTT > 0 ? (l1HTT - recoHTT) / recoHTT : outOfBounds;

  using namespace dqmoffline::l1t;
  // control plots
  fillWithinLimits(h_controlPlots_[ControlPlots::L1MET], l1MET);
  fillWithinLimits(h_controlPlots_[ControlPlots::L1ETMHF], l1ETMHF);
  fillWithinLimits(h_controlPlots_[ControlPlots::L1MHT], l1MHT);
  fillWithinLimits(h_controlPlots_[ControlPlots::L1ETT], l1ETT);
  fillWithinLimits(h_controlPlots_[ControlPlots::L1HTT], l1HTT);
  fillWithinLimits(h_controlPlots_[ControlPlots::OfflineMET], recoMET);
  fillWithinLimits(h_controlPlots_[ControlPlots::OfflineETMHF], recoETMHF);
  fillWithinLimits(h_controlPlots_[ControlPlots::OfflinePFMetNoMu], recoPFMetNoMu);
  fillWithinLimits(h_controlPlots_[ControlPlots::OfflineMHT], recoMHT);
  fillWithinLimits(h_controlPlots_[ControlPlots::OfflineETT], recoETT);
  fillWithinLimits(h_controlPlots_[ControlPlots::OfflineHTT], recoHTT);

  fill2DWithinLimits(h_L1METvsCaloMET_, recoMET, l1MET);
  fill2DWithinLimits(h_L1ETMHFvsCaloETMHF_, recoETMHF, l1ETMHF);
  fill2DWithinLimits(h_L1METvsPFMetNoMu_, recoPFMetNoMu, l1MET);
  fill2DWithinLimits(h_L1MHTvsRecoMHT_, recoMHT, l1MHT);
  fill2DWithinLimits(h_L1METTvsCaloETT_, recoETT, l1ETT);
  fill2DWithinLimits(h_L1HTTvsRecoHTT_, recoHTT, l1HTT);

  fill2DWithinLimits(h_L1METPhivsCaloMETPhi_, recoMETPhi, l1METPhi);
  fill2DWithinLimits(h_L1ETMHFPhivsCaloETMHFPhi_, recoETMHFPhi, l1ETMHFPhi);
  fill2DWithinLimits(h_L1METPhivsPFMetNoMuPhi_, recoPFMetNoMuPhi, l1METPhi);
  fill2DWithinLimits(h_L1MHTPhivsRecoMHTPhi_, recoMHTPhi, l1MHTPhi);

  fillWithinLimits(h_resolutionMET_, resolutionMET);
  fillWithinLimits(h_resolutionETMHF_, resolutionETMHF);
  fillWithinLimits(h_resolutionPFMetNoMu_, resolutionPFMetNoMu);
  fillWithinLimits(h_resolutionMHT_, resolutionMHT);
  fillWithinLimits(h_resolutionETT_, resolutionETT);
  if (resolutionMHT < outOfBounds) {
    fillWithinLimits(h_resolutionMHT_, resolutionMHT);
  }
  if (resolutionHTT < outOfBounds) {
    fillWithinLimits(h_resolutionHTT_, resolutionHTT);
  }

  fillWithinLimits(h_resolutionMETPhi_, resolutionMETPhi);
  fillWithinLimits(h_resolutionETMHFPhi_, resolutionETMHFPhi);
  fillWithinLimits(h_resolutionPFMetNoMuPhi_, resolutionPFMetNoMuPhi);
  fillWithinLimits(h_resolutionMHTPhi_, resolutionMHTPhi);

  // efficiencies
  for (auto threshold : metEfficiencyThresholds_) {
    fillWithinLimits(h_efficiencyMET_total_[threshold], recoMET);
    fillWithinLimits(h_efficiencyETMHF_total_[threshold], recoETMHF);
    fillWithinLimits(h_efficiencyPFMetNoMu_total_[threshold], recoPFMetNoMu);
    if (l1MET > threshold) {
      fillWithinLimits(h_efficiencyMET_pass_[threshold], recoMET);
      fillWithinLimits(h_efficiencyETMHF_pass_[threshold], recoETMHF);
      fillWithinLimits(h_efficiencyPFMetNoMu_pass_[threshold], recoPFMetNoMu);
    }
  }

  for (auto threshold : mhtEfficiencyThresholds_) {
    fillWithinLimits(h_efficiencyMHT_total_[threshold], recoMHT);
    if (l1MHT > threshold)
      fillWithinLimits(h_efficiencyMHT_pass_[threshold], recoMHT);
  }

  for (auto threshold : ettEfficiencyThresholds_) {
    fillWithinLimits(h_efficiencyETT_total_[threshold], recoETT);
    if (l1ETT > threshold)
      fillWithinLimits(h_efficiencyETT_pass_[threshold], recoETT);
  }

  for (auto threshold : httEfficiencyThresholds_) {
    fillWithinLimits(h_efficiencyHTT_total_[threshold], recoHTT);
    if (l1HTT > threshold)
      fillWithinLimits(h_efficiencyHTT_pass_[threshold], recoHTT);
  }
}

void L1TStage2CaloLayer2Offline::fillJets(edm::Event const& e, const unsigned int nVertex) {
  edm::Handle<l1t::JetBxCollection> l1Jets;
  e.getByToken(stage2CaloLayer2JetToken_, l1Jets);

  edm::Handle<reco::PFJetCollection> pfJets;
  e.getByToken(thePFJetCollection_, pfJets);

  if (!pfJets.isValid()) {
    edm::LogWarning("L1TStage2CaloLayer2Offline") << "invalid collection: PF jets " << std::endl;
    return;
  }
  if (!l1Jets.isValid()) {
    edm::LogWarning("L1TStage2CaloLayer2Offline") << "invalid collection: L1 jets " << std::endl;
    return;
  }

  if (pfJets->empty()) {
    LogDebug("L1TStage2CaloLayer2Offline") << "no PF jets found" << std::endl;
    return;
  }

  auto leadingRecoJet = pfJets->front();

  // find corresponding L1 jet
  double minDeltaR = 0.3;
  l1t::Jet closestL1Jet;
  bool foundMatch = false;

  //	for (int bunchCrossing = l1Jets->getFirstBX(); bunchCrossing <= l1Jets->getLastBX(); ++bunchCrossing) {
  int bunchCrossing = 0;
  for (auto jet = l1Jets->begin(bunchCrossing); jet != l1Jets->end(bunchCrossing); ++jet) {
    double currentDeltaR = deltaR(jet->eta(), jet->phi(), leadingRecoJet.eta(), leadingRecoJet.phi());
    if (currentDeltaR >= minDeltaR) {
      continue;
    } else {
      minDeltaR = currentDeltaR;
      closestL1Jet = *jet;
      foundMatch = true;
      break;
    }
  }
  //	}

  if (!foundMatch) {
    LogDebug("L1TStage2CaloLayer2Offline") << "Could not find a matching L1 Jet " << std::endl;
  }

  if (!doesNotOverlapWithHLTObjects(closestL1Jet)) {
    return;
  }

  double recoEt = leadingRecoJet.et();
  double recoEta = leadingRecoJet.eta();
  double recoPhi = leadingRecoJet.phi();

  double outOfBounds = 9999;
  double l1Et = foundMatch ? closestL1Jet.et() : 0;
  double l1Eta = foundMatch ? closestL1Jet.eta() : outOfBounds;
  double l1Phi = foundMatch ? closestL1Jet.phi() : outOfBounds;

  double resolutionEt = recoEt > 0 ? (l1Et - recoEt) / recoEt : outOfBounds;
  double resolutionEta = l1Eta - recoEta;
  double resolutionPhi = l1Phi < outOfBounds ? reco::deltaPhi(l1Phi, recoPhi) : outOfBounds;

  using namespace dqmoffline::l1t;
  // fill efficiencies regardless of matched jet found
  fillJetEfficiencies(recoEt, l1Et, recoEta);
  // control plots
  fillWithinLimits(h_controlPlots_[ControlPlots::L1JetET], l1Et);
  fillWithinLimits(h_controlPlots_[ControlPlots::OfflineJetET], recoEt);
  // don't fill anything else if no matched L1 jet is found
  if (!foundMatch) {
    return;
  }

  // eta
  fill2DWithinLimits(h_L1JetEtavsPFJetEta_, recoEta, l1Eta);
  fillWithinLimits(h_resolutionJetEta_, resolutionEta);

  if (std::abs(recoEta) <= 1.479) {  // barrel
    // et
    fill2DWithinLimits(h_L1JetETvsPFJetET_HB_, recoEt, l1Et);
    fill2DWithinLimits(h_L1JetETvsPFJetET_HB_HE_, recoEt, l1Et);
    //resolution
    fillWithinLimits(h_resolutionJetET_HB_, resolutionEt);
    fillWithinLimits(h_resolutionJetET_HB_HE_, resolutionEt);
    // phi
    fill2DWithinLimits(h_L1JetPhivsPFJetPhi_HB_, recoPhi, l1Phi);
    fill2DWithinLimits(h_L1JetPhivsPFJetPhi_HB_HE_, recoPhi, l1Phi);
    // resolution
    fillWithinLimits(h_resolutionJetPhi_HB_, resolutionPhi);
    fillWithinLimits(h_resolutionJetPhi_HB_HE_, resolutionPhi);
  } else if (std::abs(recoEta) <= 3.0) {  // end-cap
    // et
    fill2DWithinLimits(h_L1JetETvsPFJetET_HE_, recoEt, l1Et);
    fill2DWithinLimits(h_L1JetETvsPFJetET_HB_HE_, recoEt, l1Et);
    //resolution
    fillWithinLimits(h_resolutionJetET_HE_, resolutionEt);
    fillWithinLimits(h_resolutionJetET_HB_HE_, resolutionEt);
    // phi
    fill2DWithinLimits(h_L1JetPhivsPFJetPhi_HE_, recoPhi, l1Phi);
    fill2DWithinLimits(h_L1JetPhivsPFJetPhi_HB_HE_, recoPhi, l1Phi);
    // resolution
    fillWithinLimits(h_resolutionJetPhi_HE_, resolutionPhi);
    fillWithinLimits(h_resolutionJetPhi_HB_HE_, resolutionPhi);
  } else {  // forward jets
    // et
    fill2DWithinLimits(h_L1JetETvsPFJetET_HF_, recoEt, l1Et);
    // resolution
    fillWithinLimits(h_resolutionJetET_HF_, resolutionEt);
    // phi
    fill2DWithinLimits(h_L1JetPhivsPFJetPhi_HF_, recoPhi, l1Phi);
    // resolution
    fillWithinLimits(h_resolutionJetPhi_HF_, resolutionPhi);
  }
}

void L1TStage2CaloLayer2Offline::fillJetEfficiencies(const double& recoEt, const double& l1Et, const double& recoEta) {
  using namespace dqmoffline::l1t;
  if (std::abs(recoEta) <= 1.479) {  // barrel
    for (auto threshold : jetEfficiencyThresholds_) {
      fillWithinLimits(h_efficiencyJetEt_HB_total_[threshold], recoEt);
      fillWithinLimits(h_efficiencyJetEt_HB_HE_total_[threshold], recoEt);
      if (l1Et > threshold) {
        fillWithinLimits(h_efficiencyJetEt_HB_pass_[threshold], recoEt);
        fillWithinLimits(h_efficiencyJetEt_HB_HE_pass_[threshold], recoEt);
      }
    }
  } else if (std::abs(recoEta) <= 3.0) {  // end-cap
    for (auto threshold : jetEfficiencyThresholds_) {
      fillWithinLimits(h_efficiencyJetEt_HE_total_[threshold], recoEt);
      fillWithinLimits(h_efficiencyJetEt_HB_HE_total_[threshold], recoEt);
      if (l1Et > threshold) {
        fillWithinLimits(h_efficiencyJetEt_HE_pass_[threshold], recoEt);
        fillWithinLimits(h_efficiencyJetEt_HB_HE_pass_[threshold], recoEt);
      }
    }
  } else {
    for (auto threshold : jetEfficiencyThresholds_) {
      fillWithinLimits(h_efficiencyJetEt_HF_total_[threshold], recoEt);
      if (l1Et > threshold) {
        fillWithinLimits(h_efficiencyJetEt_HF_pass_[threshold], recoEt);
      }
    }  // forward jets
  }
}

//
// -------------------------------------- endRun --------------------------------------------
//
//
// -------------------------------------- book histograms --------------------------------------------
//
void L1TStage2CaloLayer2Offline::bookHistos(DQMStore::IBooker& ibooker) {
  bookEnergySumHistos(ibooker);
  bookJetHistos(ibooker);
}

void L1TStage2CaloLayer2Offline::bookEnergySumHistos(DQMStore::IBooker& ibooker) {
  ibooker.cd();
  ibooker.setCurrentFolder(histFolderEtSum_);

  dqmoffline::l1t::HistDefinition nVertexDef = histDefinitions_[PlotConfig::nVertex];
  h_nVertex_ = ibooker.book1D(nVertexDef.name, nVertexDef.title, nVertexDef.nbinsX, nVertexDef.xmin, nVertexDef.xmax);

  // energy sums control plots (monitor beyond the limits of the 2D histograms)
  h_controlPlots_[ControlPlots::L1MET] =
      ibooker.book1D("L1MET", "L1 E_{T}^{miss}; L1 E_{T}^{miss} (GeV); events", 500, -0.5, 4999.5);
  h_controlPlots_[ControlPlots::L1ETMHF] =
      ibooker.book1D("L1ETMHF", "L1 E_{T}^{miss} (HF); L1 E_{T}^{miss} (HF) (GeV); events", 500, -0.5, 4999.5);
  h_controlPlots_[ControlPlots::L1MHT] = ibooker.book1D("L1MHT", "L1 MHT; L1 MHT (GeV); events", 500, -0.5, 4999.5);
  h_controlPlots_[ControlPlots::L1ETT] = ibooker.book1D("L1ETT", "L1 ETT; L1 ETT (GeV); events", 500, -0.5, 4999.5);
  h_controlPlots_[ControlPlots::L1HTT] = ibooker.book1D("L1HTT", "L1 HTT; L1 HTT (GeV); events", 500, -0.5, 4999.5);

  h_controlPlots_[ControlPlots::OfflineMET] =
      ibooker.book1D("OfflineMET", "Offline E_{T}^{miss}; Offline E_{T}^{miss} (GeV); events", 500, -0.5, 4999.5);
  h_controlPlots_[ControlPlots::OfflineETMHF] = ibooker.book1D(
      "OfflineETMHF", "Offline E_{T}^{miss} (HF); Offline E_{T}^{miss} (HF) (GeV); events", 500, -0.5, 4999.5);
  h_controlPlots_[ControlPlots::OfflinePFMetNoMu] =
      ibooker.book1D("OfflinePFMetNoMu",
                     "Offline E_{T}^{miss} (PFMetNoMu); Offline E_{T}^{miss} (PFMetNoMu) (GeV); events",
                     500,
                     -0.5,
                     4999.5);
  h_controlPlots_[ControlPlots::OfflineMHT] =
      ibooker.book1D("OfflineMHT", "Offline MHT; Offline MHT (GeV); events", 500, -0.5, 4999.5);
  h_controlPlots_[ControlPlots::OfflineETT] =
      ibooker.book1D("OfflineETT", "Offline ETT; Offline ETT (GeV); events", 500, -0.5, 4999.5);
  h_controlPlots_[ControlPlots::OfflineHTT] =
      ibooker.book1D("OfflineHTT", "Offline HTT; Offline HTT (GeV); events", 500, -0.5, 4999.5);

  // energy sums reco vs L1
  dqmoffline::l1t::HistDefinition templateETvsET = histDefinitions_[PlotConfig::ETvsET];
  h_L1METvsCaloMET_ =
      ibooker.book2D("L1METvsCaloMET",
                     "L1 E_{T}^{miss} vs Offline E_{T}^{miss};Offline E_{T}^{miss} (GeV);L1 E_{T}^{miss} (GeV)",
                     templateETvsET.nbinsX,
                     &templateETvsET.binsX[0],
                     templateETvsET.nbinsY,
                     &templateETvsET.binsY[0]);
  h_L1ETMHFvsCaloETMHF_ = ibooker.book2D(
      "L1ETMHFvsCaloETMHF",
      "L1 E_{T}^{miss} vs Offline E_{T}^{miss} (HF);Offline E_{T}^{miss} (HF) (GeV);L1 E_{T}^{miss} (HF) (GeV)",
      templateETvsET.nbinsX,
      &templateETvsET.binsX[0],
      templateETvsET.nbinsY,
      &templateETvsET.binsY[0]);
  h_L1METvsPFMetNoMu_ = ibooker.book2D("L1METvsPFMetNoMu",
                                       "L1 E_{T}^{miss} vs Offline E_{T}^{miss} (PFMetNoMu);Offline E_{T}^{miss} "
                                       "(PFMetNoMu) (GeV);L1 E_{T}^{miss} (GeV)",
                                       templateETvsET.nbinsX,
                                       &templateETvsET.binsX[0],
                                       templateETvsET.nbinsY,
                                       &templateETvsET.binsY[0]);
  h_L1MHTvsRecoMHT_ = ibooker.book2D("L1MHTvsRecoMHT",
                                     "L1 MHT vs reco MHT;reco MHT (GeV);L1 MHT (GeV)",
                                     templateETvsET.nbinsX,
                                     &templateETvsET.binsX[0],
                                     templateETvsET.nbinsY,
                                     &templateETvsET.binsY[0]);
  h_L1METTvsCaloETT_ = ibooker.book2D("L1ETTvsCaloETT",
                                      "L1 ETT vs calo ETT;calo ETT (GeV);L1 ETT (GeV)",
                                      templateETvsET.nbinsX,
                                      &templateETvsET.binsX[0],
                                      templateETvsET.nbinsY,
                                      &templateETvsET.binsY[0]);
  h_L1HTTvsRecoHTT_ =
      ibooker.book2D("L1HTTvsRecoHTT",
                     "L1 Total H_{T} vs Offline Total H_{T};Offline Total H_{T} (GeV);L1 Total H_{T} (GeV)",
                     templateETvsET.nbinsX,
                     &templateETvsET.binsX[0],
                     templateETvsET.nbinsY,
                     &templateETvsET.binsY[0]);

  dqmoffline::l1t::HistDefinition templatePHIvsPHI = histDefinitions_[PlotConfig::PHIvsPHI];
  h_L1METPhivsCaloMETPhi_ =
      ibooker.book2D("L1METPhivsCaloMETPhi",
                     "L1 E_{T}^{miss} #phi vs Offline E_{T}^{miss} #phi;Offline E_{T}^{miss} #phi;L1 E_{T}^{miss} #phi",
                     templatePHIvsPHI.nbinsX,
                     templatePHIvsPHI.xmin,
                     templatePHIvsPHI.xmax,
                     templatePHIvsPHI.nbinsY,
                     templatePHIvsPHI.ymin,
                     templatePHIvsPHI.ymax);
  h_L1ETMHFPhivsCaloETMHFPhi_ = ibooker.book2D(
      "L1ETMHFPhivsCaloETMHFPhi",
      "L1 E_{T}^{miss} #phi vs Offline E_{T}^{miss} (HF) #phi;Offline E_{T}^{miss} (HF) #phi;L1 E_{T}^{miss} #phi",
      templatePHIvsPHI.nbinsX,
      templatePHIvsPHI.xmin,
      templatePHIvsPHI.xmax,
      templatePHIvsPHI.nbinsY,
      templatePHIvsPHI.ymin,
      templatePHIvsPHI.ymax);
  h_L1METPhivsPFMetNoMuPhi_ = ibooker.book2D("L1METPhivsPFMetNoMuPhi",
                                             "L1 E_{T}^{miss} #phi vs Offline E_{T}^{miss} (PFMetNoMu) #phi;Offline "
                                             "E_{T}^{miss} (PFMetNoMu) #phi;L1 E_{T}^{miss} #phi",
                                             templatePHIvsPHI.nbinsX,
                                             templatePHIvsPHI.xmin,
                                             templatePHIvsPHI.xmax,
                                             templatePHIvsPHI.nbinsY,
                                             templatePHIvsPHI.ymin,
                                             templatePHIvsPHI.ymax);
  h_L1MHTPhivsRecoMHTPhi_ = ibooker.book2D("L1MHTPhivsRecoMHTPhi",
                                           "L1 MHT #phi vs reco MHT #phi;reco MHT #phi;L1 MHT #phi",
                                           templatePHIvsPHI.nbinsX,
                                           templatePHIvsPHI.xmin,
                                           templatePHIvsPHI.xmax,
                                           templatePHIvsPHI.nbinsY,
                                           templatePHIvsPHI.ymin,
                                           templatePHIvsPHI.ymax);

  // energy sum resolutions
  h_resolutionMET_ =
      ibooker.book1D("resolutionMET",
                     "MET resolution; (L1 E_{T}^{miss} - Offline E_{T}^{miss})/Offline E_{T}^{miss}; events",
                     70,
                     -1.0,
                     2.5);
  h_resolutionETMHF_ =
      ibooker.book1D("resolutionETMHF",
                     "MET resolution (HF); (L1 E_{T}^{miss} - Offline E_{T}^{miss})/Offline E_{T}^{miss} (HF); events",
                     70,
                     -1.0,
                     2.5);
  h_resolutionPFMetNoMu_ = ibooker.book1D(
      "resolutionPFMetNoMu",
      "PFMetNoMu resolution; (L1 E_{T}^{miss} - Offline E_{T}^{miss})/Offline E_{T}^{miss} (PFMetNoMu); events",
      70,
      -1.0,
      2.5);
  h_resolutionMHT_ =
      ibooker.book1D("resolutionMHT", "MHT resolution; (L1 MHT - reco MHT)/reco MHT; events", 70, -1.0, 2.5);
  h_resolutionETT_ =
      ibooker.book1D("resolutionETT", "ETT resolution; (L1 ETT - calo ETT)/calo ETT; events", 70, -1.0, 2.5);
  h_resolutionHTT_ =
      ibooker.book1D("resolutionHTT",
                     "HTT resolution; (L1 Total H_{T} - Offline Total H_{T})/Offline Total H_{T}; events",
                     70,
                     -1.0,
                     2.5);

  h_resolutionMETPhi_ = ibooker.book1D(
      "resolutionMETPhi", "MET #phi resolution; (L1 E_{T}^{miss} #phi - reco MET #phi); events", 200, -1, 1);
  h_resolutionETMHFPhi_ = ibooker.book1D(
      "resolutionETMHFPhi", "MET #phi resolution (HF); (L1 E_{T}^{miss} #phi - reco MET #phi) (HF); events", 200, -1, 1);
  h_resolutionPFMetNoMuPhi_ =
      ibooker.book1D("resolutionPFMetNoMuPhi",
                     "MET #phi resolution (PFMetNoMu); (L1 E_{T}^{miss} #phi - reco MET #phi) (PFMetNoMu); events",
                     200,
                     -1,
                     1);
  h_resolutionMHTPhi_ =
      ibooker.book1D("resolutionMHTPhi", "MET #phi resolution; (L1 MHT #phi - reco MHT #phi); events", 200, -1, 1);

  // energy sum turn ons
  ibooker.setCurrentFolder(efficiencyFolderEtSum_);

  std::vector<float> metBins(metEfficiencyBins_.begin(), metEfficiencyBins_.end());
  std::vector<float> mhtBins(mhtEfficiencyBins_.begin(), mhtEfficiencyBins_.end());
  std::vector<float> ettBins(ettEfficiencyBins_.begin(), ettEfficiencyBins_.end());
  std::vector<float> httBins(httEfficiencyBins_.begin(), httEfficiencyBins_.end());

  for (auto threshold : metEfficiencyThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyMET_pass_[threshold] = ibooker.book1D("efficiencyMET_threshold_" + str_threshold + "_Num",
                                                      "MET efficiency (numerator); Offline E_{T}^{miss} (GeV);",
                                                      metBins.size() - 1,
                                                      &(metBins[0]));
    h_efficiencyMET_total_[threshold] = ibooker.book1D("efficiencyMET_threshold_" + str_threshold + "_Den",
                                                       "MET efficiency (denominator); Offline E_{T}^{miss} (GeV);",
                                                       metBins.size() - 1,
                                                       &(metBins[0]));

    h_efficiencyETMHF_pass_[threshold] = ibooker.book1D("efficiencyETMHF_threshold_" + str_threshold + "_Num",
                                                        "MET efficiency (numerator); Offline E_{T}^{miss} (GeV) (HF);",
                                                        metBins.size() - 1,
                                                        &(metBins[0]));
    h_efficiencyETMHF_total_[threshold] =
        ibooker.book1D("efficiencyETMHF_threshold_" + str_threshold + "_Den",
                       "MET efficiency (denominator); Offline E_{T}^{miss} (GeV) (HF);",
                       metBins.size() - 1,
                       &(metBins[0]));

    h_efficiencyPFMetNoMu_pass_[threshold] =
        ibooker.book1D("efficiencyPFMetNoMu_threshold_" + str_threshold + "_Num",
                       "MET efficiency (numerator); Offline E_{T}^{miss} (GeV) (PFMetNoMu);",
                       metBins.size() - 1,
                       &(metBins[0]));
    h_efficiencyPFMetNoMu_total_[threshold] =
        ibooker.book1D("efficiencyPFMetNoMu_threshold_" + str_threshold + "_Den",
                       "MET efficiency (denominator); Offline E_{T}^{miss} (GeV) (PFMetNoMu);",
                       metBins.size() - 1,
                       &(metBins[0]));
  }

  for (auto threshold : mhtEfficiencyThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyMHT_pass_[threshold] = ibooker.book1D("efficiencyMHT_threshold_" + str_threshold + "_Num",
                                                      "MHT efficiency (numerator); Offline MHT (GeV);",
                                                      mhtBins.size() - 1,
                                                      &(mhtBins[0]));
    h_efficiencyMHT_total_[threshold] = ibooker.book1D("efficiencyMHT_threshold_" + str_threshold + "_Den",
                                                       "MHT efficiency (denominator); Offline MHT (GeV);",
                                                       mhtBins.size() - 1,
                                                       &(mhtBins[0]));
  }

  for (auto threshold : ettEfficiencyThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyETT_pass_[threshold] = ibooker.book1D("efficiencyETT_threshold_" + str_threshold + "_Num",
                                                      "ETT efficiency (numerator); Offline ETT (GeV);",
                                                      ettBins.size() - 1,
                                                      &(ettBins[0]));
    h_efficiencyETT_total_[threshold] = ibooker.book1D("efficiencyETT_threshold_" + str_threshold + "_Den",
                                                       "ETT efficiency (denominator); Offline ETT (GeV);",
                                                       ettBins.size() - 1,
                                                       &(ettBins[0]));
  }
  for (auto threshold : httEfficiencyThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyHTT_pass_[threshold] = ibooker.book1D("efficiencyHTT_threshold_" + str_threshold + "_Num",
                                                      "HTT efficiency (numerator); Offline Total H_{T} (GeV);",
                                                      httBins.size() - 1,
                                                      &(httBins[0]));
    h_efficiencyHTT_total_[threshold] = ibooker.book1D("efficiencyHTT_threshold_" + str_threshold + "_Den",
                                                       "HTT efficiency (denominator); Offline Total H_{T} (GeV);",
                                                       httBins.size() - 1,
                                                       &(httBins[0]));
  }

  ibooker.cd();
}

void L1TStage2CaloLayer2Offline::bookJetHistos(DQMStore::IBooker& ibooker) {
  ibooker.cd();
  ibooker.setCurrentFolder(histFolderJet_);
  // jets control plots (monitor beyond the limits of the 2D histograms)
  h_controlPlots_[ControlPlots::L1JetET] =
      ibooker.book1D("L1JetET", "L1 Jet E_{T}; L1 Jet E_{T} (GeV); events", 500, 0, 5e3);
  h_controlPlots_[ControlPlots::OfflineJetET] =
      ibooker.book1D("OfflineJetET", "Offline Jet E_{T}; Offline Jet E_{T} (GeV); events", 500, 0, 5e3);
  // jet reco vs L1
  dqmoffline::l1t::HistDefinition templateETvsET = histDefinitions_[PlotConfig::ETvsET];
  h_L1JetETvsPFJetET_HB_ =
      ibooker.book2D("L1JetETvsPFJetET_HB",
                     "L1 Jet E_{T} vs Offline Jet E_{T} (HB); Offline Jet E_{T} (GeV); L1 Jet E_{T} (GeV)",
                     templateETvsET.nbinsX,
                     &templateETvsET.binsX[0],
                     templateETvsET.nbinsY,
                     &templateETvsET.binsY[0]);
  h_L1JetETvsPFJetET_HE_ =
      ibooker.book2D("L1JetETvsPFJetET_HE",
                     "L1 Jet E_{T} vs Offline Jet E_{T} (HE); Offline Jet E_{T} (GeV); L1 Jet E_{T} (GeV)",
                     templateETvsET.nbinsX,
                     &templateETvsET.binsX[0],
                     templateETvsET.nbinsY,
                     &templateETvsET.binsY[0]);
  h_L1JetETvsPFJetET_HF_ =
      ibooker.book2D("L1JetETvsPFJetET_HF",
                     "L1 Jet E_{T} vs Offline Jet E_{T} (HF); Offline Jet E_{T} (GeV); L1 Jet E_{T} (GeV)",
                     templateETvsET.nbinsX,
                     &templateETvsET.binsX[0],
                     templateETvsET.nbinsY,
                     &templateETvsET.binsY[0]);
  h_L1JetETvsPFJetET_HB_HE_ =
      ibooker.book2D("L1JetETvsPFJetET_HB_HE",
                     "L1 Jet E_{T} vs Offline Jet E_{T} (HB+HE); Offline Jet E_{T} (GeV); L1 Jet E_{T} (GeV)",
                     templateETvsET.nbinsX,
                     &templateETvsET.binsX[0],
                     templateETvsET.nbinsY,
                     &templateETvsET.binsY[0]);

  dqmoffline::l1t::HistDefinition templatePHIvsPHI = histDefinitions_[PlotConfig::PHIvsPHI];
  h_L1JetPhivsPFJetPhi_HB_ =
      ibooker.book2D("L1JetPhivsPFJetPhi_HB",
                     "#phi_{jet}^{L1} vs #phi_{jet}^{offline} (HB); #phi_{jet}^{offline}; #phi_{jet}^{L1}",
                     templatePHIvsPHI.nbinsX,
                     templatePHIvsPHI.xmin,
                     templatePHIvsPHI.xmax,
                     templatePHIvsPHI.nbinsY,
                     templatePHIvsPHI.ymin,
                     templatePHIvsPHI.ymax);
  h_L1JetPhivsPFJetPhi_HE_ =
      ibooker.book2D("L1JetPhivsPFJetPhi_HE",
                     "#phi_{jet}^{L1} vs #phi_{jet}^{offline} (HE); #phi_{jet}^{offline}; #phi_{jet}^{L1}",
                     templatePHIvsPHI.nbinsX,
                     templatePHIvsPHI.xmin,
                     templatePHIvsPHI.xmax,
                     templatePHIvsPHI.nbinsY,
                     templatePHIvsPHI.ymin,
                     templatePHIvsPHI.ymax);
  h_L1JetPhivsPFJetPhi_HF_ =
      ibooker.book2D("L1JetPhivsPFJetPhi_HF",
                     "#phi_{jet}^{L1} vs #phi_{jet}^{offline} (HF); #phi_{jet}^{offline}; #phi_{jet}^{L1}",
                     templatePHIvsPHI.nbinsX,
                     templatePHIvsPHI.xmin,
                     templatePHIvsPHI.xmax,
                     templatePHIvsPHI.nbinsY,
                     templatePHIvsPHI.ymin,
                     templatePHIvsPHI.ymax);
  h_L1JetPhivsPFJetPhi_HB_HE_ =
      ibooker.book2D("L1JetPhivsPFJetPhi_HB_HE",
                     "#phi_{jet}^{L1} vs #phi_{jet}^{offline} (HB+HE); #phi_{jet}^{offline}; #phi_{jet}^{L1}",
                     templatePHIvsPHI.nbinsX,
                     templatePHIvsPHI.xmin,
                     templatePHIvsPHI.xmax,
                     templatePHIvsPHI.nbinsY,
                     templatePHIvsPHI.ymin,
                     templatePHIvsPHI.ymax);

  h_L1JetEtavsPFJetEta_ = ibooker.book2D("L1JetEtavsPFJetEta_HB",
                                         "L1 Jet #eta vs Offline Jet #eta; Offline Jet #eta; L1 Jet #eta",
                                         100,
                                         -10,
                                         10,
                                         100,
                                         -10,
                                         10);

  // jet resolutions
  h_resolutionJetET_HB_ =
      ibooker.book1D("resolutionJetET_HB",
                     "jet ET resolution (HB); (L1 Jet E_{T} - Offline Jet E_{T})/Offline Jet E_{T}; events",
                     50,
                     -1,
                     1.5);
  h_resolutionJetET_HE_ =
      ibooker.book1D("resolutionJetET_HE",
                     "jet ET resolution (HE); (L1 Jet E_{T} - Offline Jet E_{T})/Offline Jet E_{T}; events",
                     50,
                     -1,
                     1.5);
  h_resolutionJetET_HF_ =
      ibooker.book1D("resolutionJetET_HF",
                     "jet ET resolution (HF); (L1 Jet E_{T} - Offline Jet E_{T})/Offline Jet E_{T}; events",
                     50,
                     -1,
                     1.5);
  h_resolutionJetET_HB_HE_ =
      ibooker.book1D("resolutionJetET_HB_HE",
                     "jet ET resolution (HB+HE); (L1 Jet E_{T} - Offline Jet E_{T})/Offline Jet E_{T}; events",
                     50,
                     -1,
                     1.5);

  h_resolutionJetPhi_HB_ =
      ibooker.book1D("resolutionJetPhi_HB",
                     "#phi_{jet} resolution (HB); (#phi_{jet}^{L1} - #phi_{jet}^{offline}); events",
                     120,
                     -0.3,
                     0.3);
  h_resolutionJetPhi_HE_ = ibooker.book1D("resolutionJetPhi_HE",
                                          "jet #phi resolution (HE); (#phi_{jet}^{L1} - #phi_{jet}^{offline}); events",
                                          120,
                                          -0.3,
                                          0.3);
  h_resolutionJetPhi_HF_ = ibooker.book1D("resolutionJetPhi_HF",
                                          "jet #phi resolution (HF); (#phi_{jet}^{L1} - #phi_{jet}^{offline}); events",
                                          120,
                                          -0.3,
                                          0.3);
  h_resolutionJetPhi_HB_HE_ =
      ibooker.book1D("resolutionJetPhi_HB_HE",
                     "jet #phi resolution (HB+HE); (#phi_{jet}^{L1} - #phi_{jet}^{offline}); events",
                     120,
                     -0.3,
                     0.3);

  h_resolutionJetEta_ = ibooker.book1D(
      "resolutionJetEta", "jet #eta resolution  (HB); (L1 Jet #eta - Offline Jet #eta); events", 120, -0.3, 0.3);

  // jet turn-ons
  ibooker.setCurrentFolder(efficiencyFolderJet_);
  std::vector<float> jetBins(jetEfficiencyBins_.begin(), jetEfficiencyBins_.end());
  int nBins = jetBins.size() - 1;
  float* jetBinArray = &(jetBins[0]);

  for (auto threshold : jetEfficiencyThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyJetEt_HB_pass_[threshold] =
        ibooker.book1D("efficiencyJetEt_HB_threshold_" + str_threshold + "_Num",
                       "jet efficiency (HB) (numerator); Offline Jet E_{T} (GeV); events",
                       nBins,
                       jetBinArray);
    h_efficiencyJetEt_HE_pass_[threshold] =
        ibooker.book1D("efficiencyJetEt_HE_threshold_" + str_threshold + "_Num",
                       "jet efficiency (HE) (numerator); Offline Jet E_{T} (GeV); events",
                       nBins,
                       jetBinArray);
    h_efficiencyJetEt_HF_pass_[threshold] =
        ibooker.book1D("efficiencyJetEt_HF_threshold_" + str_threshold + "_Num",
                       "jet efficiency (HF) (numerator); Offline Jet E_{T} (GeV); events",
                       nBins,
                       jetBinArray);
    h_efficiencyJetEt_HB_HE_pass_[threshold] =
        ibooker.book1D("efficiencyJetEt_HB_HE_threshold_" + str_threshold + "_Num",
                       "jet efficiency (HB+HE) (numerator); Offline Jet E_{T} (GeV); events",
                       nBins,
                       jetBinArray);

    h_efficiencyJetEt_HB_total_[threshold] =
        ibooker.book1D("efficiencyJetEt_HB_threshold_" + str_threshold + "_Den",
                       "jet efficiency (HB) (denominator); Offline Jet E_{T} (GeV); events",
                       nBins,
                       jetBinArray);
    h_efficiencyJetEt_HE_total_[threshold] =
        ibooker.book1D("efficiencyJetEt_HE_threshold_" + str_threshold + "_Den",
                       "jet efficiency (HE) (denominator); Offline Jet E_{T} (GeV); events",
                       nBins,
                       jetBinArray);
    h_efficiencyJetEt_HF_total_[threshold] =
        ibooker.book1D("efficiencyJetEt_HF_threshold_" + str_threshold + "_Den",
                       "jet efficiency (HF) (denominator); Offline Jet E_{T} (GeV); events",
                       nBins,
                       jetBinArray);
    h_efficiencyJetEt_HB_HE_total_[threshold] =
        ibooker.book1D("efficiencyJetEt_HB_HE_threshold_" + str_threshold + "_Den",
                       "jet efficiency (HB+HE) (denominator); Offline Jet E_{T} (GeV); events",
                       nBins,
                       jetBinArray);
  }

  ibooker.cd();
}

bool L1TStage2CaloLayer2Offline::doesNotOverlapWithHLTObjects(const l1t::Jet& jet) const {
  // get HLT objects of fired triggers
  using namespace dqmoffline::l1t;
  std::vector<bool> results = getTriggerResults(triggerIndices_, triggerResults_);
  std::vector<unsigned int> firedTriggers = getFiredTriggerIndices(triggerIndices_, results);
  std::vector<edm::InputTag> hltFilters = getHLTFilters(firedTriggers, hltConfig_, triggerProcess_);
  const trigger::TriggerObjectCollection hltObjects = getTriggerObjects(hltFilters, triggerEvent_);
  // only take objects with et() > 27 GeV
  trigger::TriggerObjectCollection filteredHltObjects;
  std::copy_if(hltObjects.begin(), hltObjects.end(), std::back_inserter(filteredHltObjects), [](auto obj) {
    return obj.et() > 27;
  });
  double l1Eta = jet.eta();
  double l1Phi = jet.phi();
  const trigger::TriggerObjectCollection matchedObjects = getMatchedTriggerObjects(l1Eta, l1Phi, 0.3, hltObjects);

  return matchedObjects.empty();
}

void L1TStage2CaloLayer2Offline::endJob() {
  // TODO: In offline, this runs after histograms are saved!
  //normalise2DHistogramsToBinArea();
}

void L1TStage2CaloLayer2Offline::normalise2DHistogramsToBinArea() {
  std::vector<MonitorElement*> monElementstoNormalize = {
      h_L1METvsCaloMET_,         h_L1ETMHFvsCaloETMHF_,       h_L1METvsPFMetNoMu_,      h_L1MHTvsRecoMHT_,
      h_L1METTvsCaloETT_,        h_L1HTTvsRecoHTT_,           h_L1METPhivsCaloMETPhi_,  h_L1ETMHFPhivsCaloETMHFPhi_,
      h_L1METPhivsPFMetNoMuPhi_, h_L1MHTPhivsRecoMHTPhi_,     h_L1JetETvsPFJetET_HB_,   h_L1JetETvsPFJetET_HE_,
      h_L1JetETvsPFJetET_HF_,    h_L1JetETvsPFJetET_HB_HE_,   h_L1JetPhivsPFJetPhi_HB_, h_L1JetPhivsPFJetPhi_HE_,
      h_L1JetPhivsPFJetPhi_HF_,  h_L1JetPhivsPFJetPhi_HB_HE_, h_L1JetEtavsPFJetEta_,
  };

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
DEFINE_FWK_MODULE(L1TStage2CaloLayer2Offline);
