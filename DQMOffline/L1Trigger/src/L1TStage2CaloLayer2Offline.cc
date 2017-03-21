#include "DQMOffline/L1Trigger/interface/L1TStage2CaloLayer2Offline.h"
#include "DQMOffline/L1Trigger/interface/L1TFillWithinLimits.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
// Geometry
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "TLorentzVector.h"

//
// -------------------------------------- Constructor --------------------------------------------
//
L1TStage2CaloLayer2Offline::L1TStage2CaloLayer2Offline(const edm::ParameterSet& ps) :
        theCaloJetCollection_(
            consumes < reco::CaloJetCollection > (ps.getParameter < edm::InputTag > ("caloJetCollection"))),
        thecaloMETCollection_(
            consumes < reco::CaloMETCollection > (ps.getParameter < edm::InputTag > ("caloMETCollection"))),
        thePVCollection_(consumes < reco::VertexCollection > (ps.getParameter < edm::InputTag > ("PVCollection"))),
        theBSCollection_(consumes < reco::BeamSpot > (ps.getParameter < edm::InputTag > ("beamSpotCollection"))),
        triggerEvent_(consumes < trigger::TriggerEvent > (ps.getParameter < edm::InputTag > ("TriggerEvent"))),
        triggerResults_(consumes < edm::TriggerResults > (ps.getParameter < edm::InputTag > ("TriggerResults"))),
        triggerFilter_(ps.getParameter < edm::InputTag > ("TriggerFilter")),
        triggerPath_(ps.getParameter < std::string > ("TriggerPath")),
        histFolder_(ps.getParameter < std::string > ("histFolder")),
        efficiencyFolder_(histFolder_ + "/efficiency_raw"),
        stage2CaloLayer2JetToken_(
            consumes < l1t::JetBxCollection > (ps.getParameter < edm::InputTag > ("stage2CaloLayer2JetSource"))),
        stage2CaloLayer2EtSumToken_(
            consumes < l1t::EtSumBxCollection > (ps.getParameter < edm::InputTag > ("stage2CaloLayer2EtSumSource"))),
        jetEfficiencyThresholds_(ps.getParameter < std::vector<double> > ("jetEfficiencyThresholds")),
        metEfficiencyThresholds_(ps.getParameter < std::vector<double> > ("metEfficiencyThresholds")),
        mhtEfficiencyThresholds_(ps.getParameter < std::vector<double> > ("mhtEfficiencyThresholds")),
        ettEfficiencyThresholds_(ps.getParameter < std::vector<double> > ("ettEfficiencyThresholds")),
        httEfficiencyThresholds_(ps.getParameter < std::vector<double> > ("httEfficiencyThresholds")),
        jetEfficiencyBins_(ps.getParameter < std::vector<double> > ("jetEfficiencyBins")),
        metEfficiencyBins_(ps.getParameter < std::vector<double> > ("metEfficiencyBins")),
        mhtEfficiencyBins_(ps.getParameter < std::vector<double> > ("mhtEfficiencyBins")),
        ettEfficiencyBins_(ps.getParameter < std::vector<double> > ("ettEfficiencyBins")),
        httEfficiencyBins_(ps.getParameter < std::vector<double> > ("httEfficiencyBins"))
{
  edm::LogInfo("L1TStage2CaloLayer2Offline") << "Constructor "
      << "L1TStage2CaloLayer2Offline::L1TStage2CaloLayer2Offline " << std::endl;
}

//
// -- Destructor
//
L1TStage2CaloLayer2Offline::~L1TStage2CaloLayer2Offline()
{
  edm::LogInfo("L1TStage2CaloLayer2Offline") << "Destructor L1TStage2CaloLayer2Offline::~L1TStage2CaloLayer2Offline "
      << std::endl;
}

//
// -------------------------------------- beginRun --------------------------------------------
//
void L1TStage2CaloLayer2Offline::dqmBeginRun(edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("L1TStage2CaloLayer2Offline") << "L1TStage2CaloLayer2Offline::beginRun" << std::endl;
}
//
// -------------------------------------- bookHistos --------------------------------------------
//
void L1TStage2CaloLayer2Offline::bookHistograms(DQMStore::IBooker & ibooker_, edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("L1TStage2CaloLayer2Offline") << "L1TStage2CaloLayer2Offline::bookHistograms" << std::endl;

  //book at beginRun
  bookHistos(ibooker_);
}
//
// -------------------------------------- beginLuminosityBlock --------------------------------------------
//
void L1TStage2CaloLayer2Offline::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
    edm::EventSetup const& context)
{
  edm::LogInfo("L1TStage2CaloLayer2Offline") << "L1TStage2CaloLayer2Offline::beginLuminosityBlock" << std::endl;
}

//
// -------------------------------------- Analyze --------------------------------------------
//
void L1TStage2CaloLayer2Offline::analyze(edm::Event const& e, edm::EventSetup const& eSetup)
{
  edm::LogInfo("L1TStage2CaloLayer2Offline") << "L1TStage2CaloLayer2Offline::analyze" << std::endl;

  edm::Handle<reco::VertexCollection> vertexHandle;
  e.getByToken(thePVCollection_, vertexHandle);
  if (!vertexHandle.isValid()) {
    edm::LogError("L1TStage2CaloLayer2Offline") << "invalid collection: vertex " << std::endl;
    return;
  }

  unsigned int nVertex = vertexHandle->size();
  dqmoffline::l1t::fillWithinLimits(h_nVertex_, nVertex);

  // L1T
  fillEnergySums(e, nVertex);
  fillJets(e, nVertex);
}

void L1TStage2CaloLayer2Offline::fillEnergySums(edm::Event const& e, const unsigned int nVertex)
{
  edm::Handle<l1t::EtSumBxCollection> l1EtSums;
  e.getByToken(stage2CaloLayer2EtSumToken_, l1EtSums);

  edm::Handle<reco::CaloJetCollection> caloJets;
  e.getByToken(theCaloJetCollection_, caloJets);

  edm::Handle<reco::CaloMETCollection> caloMETs;
  e.getByToken(thecaloMETCollection_, caloMETs);

  if (!caloJets.isValid()) {
    edm::LogError("L1TStage2CaloLayer2Offline") << "invalid collection: calo jets " << std::endl;
    return;
  }
  if (!caloMETs.isValid()) {
    edm::LogError("L1TStage2CaloLayer2Offline") << "invalid collection: Offline E_{T}^{miss} " << std::endl;
    return;
  }
  if (!l1EtSums.isValid()) {
    edm::LogError("L1TStage2CaloLayer2Offline") << "invalid collection: L1 ET sums " << std::endl;
    return;
  }

  int bunchCrossing = 0;

  double l1MET(0);
  double l1METPhi(0);
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
  double recoMHT(0);
  double recoMHTPhi(0);
  double recoETT(0);
  double recoHTT(0);

  TVector2 mht(0., 0.);

  for (auto jet = caloJets->begin(); jet != caloJets->end(); ++jet) {
    double et = jet->et();
    if (et < 30) {
      continue;
    }
    TVector2 jetVec(et * cos(jet->phi()), et * sin(jet->phi()));
    recoHTT += et;
    mht -= jetVec;
  }
  recoETT = recoHTT;
  recoMHT = mht.Mod();
  // phi in cms is defined between -pi and pi
  recoMHTPhi = TVector2::Phi_mpi_pi(mht.Phi());

  // if no reco value, relative resolution does not make sense -> sort to overflow
  double outOfBounds = 9999;
  double resolutionMET = recoMET > 0 ? (l1MET - recoMET) / recoMET : outOfBounds;
  double resolutionMETPhi = std::abs(recoMETPhi) > 0 ? (l1METPhi - recoMETPhi) / recoMETPhi : outOfBounds;
  double resolutionMHT = recoMHT > 0 ? (l1MHT - recoMHT) / recoMHT : outOfBounds;
  double resolutionMHTPhi = std::abs(recoMHTPhi) > 0 ? (l1MHTPhi - recoMHTPhi) / recoMHTPhi : outOfBounds;
  double resolutionETT = recoETT > 0 ? (l1ETT - recoETT) / recoETT : outOfBounds;
  double resolutionHTT = recoHTT > 0 ? (l1HTT - recoHTT) / recoHTT : outOfBounds;

  using namespace dqmoffline::l1t;
  fill2DWithinLimits(h_L1METvsCaloMET_, recoMET, l1MET);
  fill2DWithinLimits(h_L1MHTvsRecoMHT_, recoMHT, l1MHT);
  fill2DWithinLimits(h_L1METTvsCaloETT_, recoETT, l1ETT);
  fill2DWithinLimits(h_L1HTTvsRecoHTT_, recoHTT, l1HTT);

  fill2DWithinLimits(h_L1METPhivsCaloMETPhi_, recoMETPhi, l1METPhi);
  fill2DWithinLimits(h_L1MHTPhivsRecoMHTPhi_, recoMHTPhi, l1MHTPhi);

  fillWithinLimits(h_resolutionMET_, resolutionMET);
  fillWithinLimits(h_resolutionMHT_, resolutionMHT);
  fillWithinLimits(h_resolutionETT_, resolutionETT);
  fillWithinLimits(h_resolutionHTT_, resolutionHTT);

  fillWithinLimits(h_resolutionMETPhi_, resolutionMETPhi);
  fillWithinLimits(h_resolutionMHTPhi_, resolutionMHTPhi);

  // efficiencies
  for (auto threshold : metEfficiencyThresholds_) {
    fillWithinLimits(h_efficiencyMET_total_[threshold], recoMET);
    if (l1MET > threshold)
      fillWithinLimits(h_efficiencyMET_pass_[threshold], recoMET);
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

void L1TStage2CaloLayer2Offline::fillJets(edm::Event const& e, const unsigned int nVertex)
{
  edm::Handle<l1t::JetBxCollection> l1Jets;
  e.getByToken(stage2CaloLayer2JetToken_, l1Jets);

  edm::Handle<reco::CaloJetCollection> caloJets;
  e.getByToken(theCaloJetCollection_, caloJets);

  if (!caloJets.isValid()) {
    edm::LogError("L1TStage2CaloLayer2Offline") << "invalid collection: calo jets " << std::endl;
    return;
  }
  if (!l1Jets.isValid()) {
    edm::LogError("L1TStage2CaloLayer2Offline") << "invalid collection: L1 jets " << std::endl;
    return;
  }

  if (caloJets->size() == 0) {
    LogDebug("L1TStage2CaloLayer2Offline") << "no calo jets found" << std::endl;
    return;
  }

  if (l1Jets->size() == 0) {
    LogDebug("L1TStage2CaloLayer2Offline") << "no L1 jets found" << std::endl;
    return;
  }

  auto leadingRecoJet = caloJets->front();

  // find corresponding L1 jet
  double minDeltaR = 0.3;
  l1t::Jet closestL1Jet;
  bool foundMatch = false;

//	for (int bunchCrossing = l1Jets->getFirstBX(); bunchCrossing <= l1Jets->getLastBX(); ++bunchCrossing) {
  int bunchCrossing = 0;
  for (auto jet = l1Jets->begin(bunchCrossing); jet != l1Jets->end(bunchCrossing); ++jet) {
    double currentDeltaR = deltaR(jet->eta(), jet->phi(), leadingRecoJet.eta(), leadingRecoJet.phi());
    if (currentDeltaR > minDeltaR) {
      continue;
    } else {
      minDeltaR = currentDeltaR;
      closestL1Jet = *jet;
      foundMatch = true;
    }

  }
//	}

  if (!foundMatch) {
    edm::LogError("L1TStage2CaloLayer2Offline") << "Could not find a matching L1 Jet " << std::endl;
    return;
  }

  double recoEt = leadingRecoJet.et();
  double recoEta = leadingRecoJet.eta();
  double recoPhi = leadingRecoJet.phi();

  double l1Et = closestL1Jet.et();
  double l1Eta = closestL1Jet.eta();
  double l1Phi = closestL1Jet.phi();

  // if no reco value, relative resolution does not make sense -> sort to overflow
  double outOfBounds = 9999;
  double resolutionEt = recoEt > 0 ? (l1Et - recoEt) / recoEt : outOfBounds;
  double resolutionEta = std::abs(recoEta) > 0 ? (l1Eta - recoEta) / recoEta : outOfBounds;
  double resolutionPhi = std::abs(recoPhi) > 0 ? (l1Phi - recoPhi) / recoPhi : outOfBounds;

  using namespace dqmoffline::l1t;
  // eta
  fill2DWithinLimits(h_L1JetEtavsCaloJetEta_, recoEta, l1Eta);
  fillWithinLimits(h_resolutionJetEta_, resolutionEta);

  if (std::abs(recoEta) <= 1.479) { // barrel
    // et
    fill2DWithinLimits(h_L1JetETvsCaloJetET_HB_, recoEt, l1Et);
    fill2DWithinLimits(h_L1JetETvsCaloJetET_HB_HE_, recoEt, l1Et);
    //resolution
    fillWithinLimits(h_resolutionJetET_HB_, resolutionEt);
    fillWithinLimits(h_resolutionJetET_HB_HE_, resolutionEt);
    // phi
    fill2DWithinLimits(h_L1JetPhivsCaloJetPhi_HB_, recoPhi, l1Phi);
    fill2DWithinLimits(h_L1JetPhivsCaloJetPhi_HB_HE_, recoPhi, l1Phi);
    // resolution
    fillWithinLimits(h_resolutionJetPhi_HB_, resolutionPhi);
    fillWithinLimits(h_resolutionJetPhi_HB_HE_, resolutionPhi);

    // turn-ons

    for (auto threshold : jetEfficiencyThresholds_) {
      fillWithinLimits(h_efficiencyJetEt_HB_total_[threshold], recoEt);
      fillWithinLimits(h_efficiencyJetEt_HB_HE_total_[threshold], recoEt);
      if (l1Et > threshold) {
        fillWithinLimits(h_efficiencyJetEt_HB_pass_[threshold], recoEt);
        fillWithinLimits(h_efficiencyJetEt_HB_HE_pass_[threshold], recoEt);
      }
    }

  } else if (std::abs(recoEta) <= 3.0) { // end-cap
    // et
    fill2DWithinLimits(h_L1JetETvsCaloJetET_HE_, recoEt, l1Et);
    fill2DWithinLimits(h_L1JetETvsCaloJetET_HB_HE_, recoEt, l1Et);
    //resolution
    fillWithinLimits(h_resolutionJetET_HE_, resolutionEt);
    fillWithinLimits(h_resolutionJetET_HB_HE_, resolutionEt);
    // phi
    fill2DWithinLimits(h_L1JetPhivsCaloJetPhi_HE_, recoPhi, l1Phi);
    fill2DWithinLimits(h_L1JetPhivsCaloJetPhi_HB_HE_, recoPhi, l1Phi);
    // resolution
    fillWithinLimits(h_resolutionJetPhi_HE_, resolutionPhi);
    fillWithinLimits(h_resolutionJetPhi_HB_HE_, resolutionPhi);

    // turn-ons
    for (auto threshold : jetEfficiencyThresholds_) {
      fillWithinLimits(h_efficiencyJetEt_HE_total_[threshold], recoEt);
      fillWithinLimits(h_efficiencyJetEt_HB_HE_total_[threshold], recoEt);
      if (l1Et > threshold) {
        fillWithinLimits(h_efficiencyJetEt_HE_pass_[threshold], recoEt);
        fillWithinLimits(h_efficiencyJetEt_HB_HE_pass_[threshold], recoEt);
      }
    }
  } else { // forward jets
    // et
    fill2DWithinLimits(h_L1JetETvsCaloJetET_HF_, recoEt, l1Et);
    // resolution
    fillWithinLimits(h_resolutionJetET_HF_, resolutionEt);
    // phi
    fill2DWithinLimits(h_L1JetPhivsCaloJetPhi_HF_, recoPhi, l1Phi);
    // resolution
    fillWithinLimits(h_resolutionJetPhi_HF_, resolutionPhi);
    // turn-ons
    for (auto threshold : jetEfficiencyThresholds_) {
      fillWithinLimits(h_efficiencyJetEt_HF_total_[threshold], recoEt);
      if (l1Et > threshold) {
        fillWithinLimits(h_efficiencyJetEt_HF_pass_[threshold], recoEt);
      }
    }
  }
}

//
// -------------------------------------- endLuminosityBlock --------------------------------------------
//
void L1TStage2CaloLayer2Offline::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup)
{
  edm::LogInfo("L1TStage2CaloLayer2Offline") << "L1TStage2CaloLayer2Offline::endLuminosityBlock" << std::endl;
}

//
// -------------------------------------- endRun --------------------------------------------
//
void L1TStage2CaloLayer2Offline::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  edm::LogInfo("L1TStage2CaloLayer2Offline") << "L1TStage2CaloLayer2Offline::endRun" << std::endl;
}

//
// -------------------------------------- book histograms --------------------------------------------
//
void L1TStage2CaloLayer2Offline::bookHistos(DQMStore::IBooker & ibooker)
{
  bookEnergySumHistos(ibooker);
  bookJetHistos(ibooker);
}

void L1TStage2CaloLayer2Offline::bookEnergySumHistos(DQMStore::IBooker & ibooker)
{
  ibooker.cd();
  ibooker.setCurrentFolder(histFolder_.c_str());

  h_nVertex_ = ibooker.book1D("nVertex", "Number of event vertices in collection", 40, -0.5, 39.5);

  // energy sums reco vs L1
  h_L1METvsCaloMET_ = ibooker.book2D("L1METvsCaloMET",
      "L1 E_{T}^{miss} vs Offline E_{T}^{miss};Offline E_{T}^{miss} (GeV);L1 E_{T}^{miss} (GeV)", 500, -0.5, 499.5, 500,
      -0.5, 499.5);
  h_L1MHTvsRecoMHT_ = ibooker.book2D("L1MHTvsRecoMHT", "L1 MHT vs reco MHT;reco MHT (GeV);L1 MHT (GeV)", 500, -0.5,
      499.5, 500, -0.5, 499.5);
  h_L1METTvsCaloETT_ = ibooker.book2D("L1ETTvsCaloETT", "L1 ETT vs calo ETT;calo ETT (GeV);L1 ETT (GeV)", 500, -0.5,
      499.5, 500, -0.5, 499.5);
  h_L1HTTvsRecoHTT_ = ibooker.book2D("L1HTTvsRecoHTT",
      "L1 Total H_{T} vs Offline Total H_{T};Offline Total H_{T} (GeV);L1 Total H_{T} (GeV)", 500, -0.5, 499.5, 500,
      -0.5, 499.5);

  h_L1METPhivsCaloMETPhi_ = ibooker.book2D("L1METPhivsCaloMETPhi",
      "L1 E_{T}^{miss} #phi vs Offline E_{T}^{miss} #phi;Offline E_{T}^{miss} #phi;L1 E_{T}^{miss} #phi", 100, -4, 4,
      100, -4, 4);
  h_L1MHTPhivsRecoMHTPhi_ = ibooker.book2D("L1MHTPhivsRecoMHTPhi",
      "L1 MHT #phi vs reco MHT #phi;reco MHT #phi;L1 MHT #phi", 100, -4, 4, 100, -4, 4);

  // energy sum resolutions
  h_resolutionMET_ = ibooker.book1D("resolutionMET",
      "MET resolution; (L1 E_{T}^{miss} - Offline E_{T}^{miss})/Offline E_{T}^{miss}; events", 50, -1, 1.5);
  h_resolutionMHT_ = ibooker.book1D("resolutionMHT", "MHT resolution; (L1 MHT - reco MHT)/reco MHT; events", 50, -1,
      1.5);
  h_resolutionETT_ = ibooker.book1D("resolutionETT", "ETT resolution; (L1 ETT - calo ETT)/calo ETT; events", 50, -1,
      1.5);
  h_resolutionHTT_ = ibooker.book1D("resolutionHTT",
      "HTT resolution; (L1 Total H_{T} - Offline Total H_{T})/Offline Total H_{T}; events", 50, -1, 1.5);

  h_resolutionMETPhi_ = ibooker.book1D("resolutionMETPhi",
      "MET #phi resolution; (L1 E_{T}^{miss} #phi - reco MET #phi)/reco MET #phi; events", 120, -0.3, 0.3);
  h_resolutionMHTPhi_ = ibooker.book1D("resolutionMHTPhi",
      "MET #phi resolution; (L1 MHT #phi - reco MHT #phi)/reco MHT #phi; events", 120, -0.3, 0.3);

  // energy sum turn ons
  ibooker.setCurrentFolder(efficiencyFolder_.c_str());

  std::vector<float> metBins(metEfficiencyBins_.begin(), metEfficiencyBins_.end());
  std::vector<float> mhtBins(mhtEfficiencyBins_.begin(), mhtEfficiencyBins_.end());
  std::vector<float> ettBins(ettEfficiencyBins_.begin(), ettEfficiencyBins_.end());
  std::vector<float> httBins(httEfficiencyBins_.begin(), httEfficiencyBins_.end());

  for (auto threshold : metEfficiencyThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyMET_pass_[threshold] = ibooker.book1D("efficiencyMET_threshold_" + str_threshold + "_Num",
        "MET efficiency; Offline E_{T}^{miss} (GeV); events", metBins.size() - 1, &(metBins[0]));
    h_efficiencyMET_total_[threshold] = ibooker.book1D("efficiencyMET_threshold_" + str_threshold + "_Den",
        "MET efficiency; Offline E_{T}^{miss} (GeV); events", metBins.size() - 1, &(metBins[0]));
  }

  for (auto threshold : mhtEfficiencyThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyMHT_pass_[threshold] = ibooker.book1D("efficiencyMHT_threshold_" + str_threshold + "_Num",
        "MHT efficiency; Offline MHT (GeV); events", mhtBins.size() - 1, &(mhtBins[0]));
    h_efficiencyMHT_total_[threshold] = ibooker.book1D("efficiencyMHT_threshold_" + str_threshold + "_Den",
        "MHT efficiency; Offline MHT (GeV); events", mhtBins.size() - 1, &(mhtBins[0]));
  }

  for (auto threshold : ettEfficiencyThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyETT_pass_[threshold] = ibooker.book1D("efficiencyETT_threshold_" + str_threshold + "_Num",
        "ETT efficiency; Offline ETT (GeV); events", ettBins.size() - 1, &(ettBins[0]));
    h_efficiencyETT_total_[threshold] = ibooker.book1D("efficiencyETT_threshold_" + str_threshold + "_Den",
        "ETT efficiency; Offline ETT (GeV); events", ettBins.size() - 1, &(ettBins[0]));
  }
  for (auto threshold : httEfficiencyThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyHTT_pass_[threshold] = ibooker.book1D("efficiencyHTT_threshold_" + str_threshold + "_Num",
        "HTT efficiency; Offline Total H_{T} (GeV); events", httBins.size() - 1, &(httBins[0]));
    h_efficiencyHTT_total_[threshold] = ibooker.book1D("efficiencyHTT_threshold_" + str_threshold + "_Den",
        "HTT efficiency; Offline Total H_{T} (GeV); events", httBins.size() - 1, &(httBins[0]));
  }

  ibooker.cd();
}

void L1TStage2CaloLayer2Offline::bookJetHistos(DQMStore::IBooker & ibooker)
{
  ibooker.cd();
  ibooker.setCurrentFolder(histFolder_.c_str());
  // jet reco vs L1
  h_L1JetETvsCaloJetET_HB_ = ibooker.book2D("L1JetETvsCaloJetET_HB",
      "L1 Jet E_{T} vs Offline Jet E_{T} (HB); Offline Jet E_{T} (GeV); L1 Jet E_{T} (GeV)", 300, 0, 300, 300, 0, 300);
  h_L1JetETvsCaloJetET_HE_ = ibooker.book2D("L1JetETvsCaloJetET_HE",
      "L1 Jet E_{T} vs Offline Jet E_{T} (HE); Offline Jet E_{T} (GeV); L1 Jet E_{T} (GeV)", 300, 0, 300, 300, 0, 300);
  h_L1JetETvsCaloJetET_HF_ = ibooker.book2D("L1JetETvsCaloJetET_HF",
      "L1 Jet E_{T} vs Offline Jet E_{T} (HF); Offline Jet E_{T} (GeV); L1 Jet E_{T} (GeV)", 300, 0, 300, 300, 0, 300);
  h_L1JetETvsCaloJetET_HB_HE_ = ibooker.book2D("L1JetETvsCaloJetET_HB_HE",
      "L1 Jet E_{T} vs Offline Jet E_{T} (HB+HE); Offline Jet E_{T} (GeV); L1 Jet E_{T} (GeV)", 300, 0, 300, 300, 0,
      300);

  h_L1JetPhivsCaloJetPhi_HB_ = ibooker.book2D("L1JetETvsCaloJetET_HB",
      "#phi_{jet}^{L1} vs #phi_{jet}^{offline} (HB); #phi_{jet}^{offline}; #phi_{jet}^{L1}", 100, -4, 4, 100, -4, 4);
  h_L1JetPhivsCaloJetPhi_HE_ = ibooker.book2D("L1JetETvsCaloJetET_HE",
      "#phi_{jet}^{L1} vs #phi_{jet}^{offline} (HE); #phi_{jet}^{offline}; #phi_{jet}^{L1}", 100, -4, 4, 100, -4, 4);
  h_L1JetPhivsCaloJetPhi_HF_ = ibooker.book2D("L1JetETvsCaloJetET_HF",
      "#phi_{jet}^{L1} vs #phi_{jet}^{offline} (HF); #phi_{jet}^{offline}; #phi_{jet}^{L1}", 100, -4, 4, 100, -4, 4);
  h_L1JetPhivsCaloJetPhi_HB_HE_ = ibooker.book2D("L1JetETvsCaloJetET_HB_HE",
      "#phi_{jet}^{L1} vs #phi_{jet}^{offline} (HB+HE); #phi_{jet}^{offline}; #phi_{jet}^{L1}", 100, -4, 4, 100, -4, 4);

  h_L1JetEtavsCaloJetEta_ = ibooker.book2D("L1JetEtavsCaloJetEta_HB",
      "L1 Jet #eta vs Offline Jet #eta; Offline Jet #eta; L1 Jet #eta", 100, -10, 10, 100, -10, 10);

  // jet resolutions
  h_resolutionJetET_HB_ = ibooker.book1D("resolutionJetET_HB",
      "jet ET resolution (HB); (L1 Jet E_{T} - Offline Jet E_{T})/Offline Jet E_{T}; events", 50, -1, 1.5);
  h_resolutionJetET_HE_ = ibooker.book1D("resolutionJetET_HE",
      "jet ET resolution (HE); (L1 Jet E_{T} - Offline Jet E_{T})/Offline Jet E_{T}; events", 50, -1, 1.5);
  h_resolutionJetET_HF_ = ibooker.book1D("resolutionJetET_HF",
      "jet ET resolution (HF); (L1 Jet E_{T} - Offline Jet E_{T})/Offline Jet E_{T}; events", 50, -1, 1.5);
  h_resolutionJetET_HB_HE_ = ibooker.book1D("resolutionJetET_HB_HE",
      "jet ET resolution (HB+HE); (L1 Jet E_{T} - Offline Jet E_{T})/Offline Jet E_{T}; events", 50, -1, 1.5);

  h_resolutionJetPhi_HB_ = ibooker.book1D("resolutionJetPhi_HB",
      "#phi_{jet} resolution (HB); (#phi_{jet}^{L1} - #phi_{jet}^{offline})/#phi_{jet}^{offline}; events", 120, -0.3,
      0.3);
  h_resolutionJetPhi_HE_ = ibooker.book1D("resolutionJetPhi_HE",
      "jet #phi resolution (HE); (#phi_{jet}^{L1} - #phi_{jet}^{offline})/#phi_{jet}^{offline}; events", 120, -0.3,
      0.3);
  h_resolutionJetPhi_HF_ = ibooker.book1D("resolutionJetPhi_HF",
      "jet #phi resolution (HF); (#phi_{jet}^{L1} - #phi_{jet}^{offline})/#phi_{jet}^{offline}; events", 120, -0.3,
      0.3);
  h_resolutionJetPhi_HB_HE_ = ibooker.book1D("resolutionJetPhi_HB_HE",
      "jet #phi resolution (HB+HE); (#phi_{jet}^{L1} - #phi_{jet}^{offline})/#phi_{jet}^{offline}; events", 120, -0.3,
      0.3);

  h_resolutionJetEta_ = ibooker.book1D("resolutionJetEta",
      "jet #eta resolution  (HB); (L1 Jet #eta - Offline Jet #eta)/Offline Jet #eta; events", 120, -0.3, 0.3);

  // jet turn-ons
  ibooker.setCurrentFolder(efficiencyFolder_.c_str());
  std::vector<float> jetBins(jetEfficiencyBins_.begin(), jetEfficiencyBins_.end());
  int nBins = jetBins.size() - 1;
  float* jetBinArray = &(jetBins[0]);

  for (auto threshold : jetEfficiencyThresholds_) {
    std::string str_threshold = std::to_string(int(threshold));
    h_efficiencyJetEt_HB_pass_[threshold] = ibooker.book1D("efficiencyJetEt_HB_threshold_" + str_threshold + "_Num",
        "jet efficiency (HB); Offline Jet E_{T} (GeV); events", nBins, jetBinArray);
    h_efficiencyJetEt_HE_pass_[threshold] = ibooker.book1D("efficiencyJetEt_HE_threshold_" + str_threshold + "_Num",
        "jet efficiency (HE); Offline Jet E_{T} (GeV); events", nBins, jetBinArray);
    h_efficiencyJetEt_HF_pass_[threshold] = ibooker.book1D("efficiencyJetEt_HF_threshold_" + str_threshold + "_Num",
        "jet efficiency (HF); Offline Jet E_{T} (GeV); events", nBins, jetBinArray);
    h_efficiencyJetEt_HB_HE_pass_[threshold] = ibooker.book1D(
        "efficiencyJetEt_HB_HE_threshold_" + str_threshold + "_Num",
        "jet efficiency (HB+HE); Offline Jet E_{T} (GeV); events", nBins, jetBinArray);

    h_efficiencyJetEt_HB_total_[threshold] = ibooker.book1D("efficiencyJetEt_HB_threshold_" + str_threshold + "_Den",
        "jet efficiency (HB); Offline Jet E_{T} (GeV); events", nBins, jetBinArray);
    h_efficiencyJetEt_HE_total_[threshold] = ibooker.book1D("efficiencyJetEt_HE_threshold_" + str_threshold + "_Den",
        "jet efficiency (HE); Offline Jet E_{T} (GeV); events", nBins, jetBinArray);
    h_efficiencyJetEt_HF_total_[threshold] = ibooker.book1D("efficiencyJetEt_HF_threshold_" + str_threshold + "_Den",
        "jet efficiency (HF); Offline Jet E_{T} (GeV); events", nBins, jetBinArray);
    h_efficiencyJetEt_HB_HE_total_[threshold] = ibooker.book1D(
        "efficiencyJetEt_HB_HE_threshold_" + str_threshold + "_Den",
        "jet efficiency (HB+HE); Offline Jet E_{T} (GeV); events", nBins, jetBinArray);
  }

  ibooker.cd();
}

//define this as a plug-in
DEFINE_FWK_MODULE (L1TStage2CaloLayer2Offline);
