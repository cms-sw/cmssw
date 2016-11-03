#include "DQMOffline/L1Trigger/interface/L1TStage2CaloLayer2Offline.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
// Geometry
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "TLorentzVector.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

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
        stage2CaloLayer2JetToken_(
            consumes < l1t::JetBxCollection > (ps.getParameter < edm::InputTag > ("stage2CaloLayer2JetSource"))),
        stage2CaloLayer2EtSumToken_(
            consumes < l1t::EtSumBxCollection > (ps.getParameter < edm::InputTag > ("stage2CaloLayer2EtSumSource")))
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
  h_nVertex_->Fill(nVertex);

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
    edm::LogError("L1TStage2CaloLayer2Offline") << "invalid collection: calo MET " << std::endl;
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
    TVector2 jetVec(0., 0.);
    jetVec.SetMagPhi(et, jet->phi());
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
  double resolutionMETPhi = abs(recoMETPhi) > 0 ? (l1METPhi - recoMETPhi) / recoMETPhi : outOfBounds;
  double resolutionMHT = recoMHT > 0 ? (l1MHT - recoMHT) / recoMHT : outOfBounds;
  double resolutionMHTPhi = abs(recoMHTPhi) > 0 ? (l1MHTPhi - recoMHTPhi) / recoMHTPhi : outOfBounds;
  double resolutionETT = recoETT > 0 ? (l1ETT - recoETT) / recoETT : outOfBounds;
  double resolutionHTT = recoHTT > 0 ? (l1HTT - recoHTT) / recoHTT : outOfBounds;

  h_L1METvsCaloMET_->Fill(recoMET, l1MET);
  h_L1MHTvsRecoMHT_->Fill(recoMHT, l1MHT);
  h_L1METTvsCaloETT_->Fill(recoETT, l1ETT);
  h_L1HTTvsRecoHTT_->Fill(recoHTT, l1HTT);

  h_L1METPhivsCaloMETPhi_->Fill(recoMETPhi, l1METPhi);
  h_L1MHTPhivsRecoMHTPhi_->Fill(recoMHTPhi, l1MHTPhi);

  h_resolutionMET_->Fill(resolutionMET);
  h_resolutionMHT_->Fill(resolutionMHT);
  h_resolutionETT_->Fill(resolutionETT);
  h_resolutionHTT_->Fill(resolutionHTT);

  h_resolutionMETPhi_->Fill(resolutionMETPhi);
  h_resolutionMHTPhi_->Fill(resolutionMHTPhi);

  h_efficiencyMET_total_->Fill(recoMET);
  h_efficiencyMHT_total_->Fill(recoMHT);
  h_efficiencyETT_total_->Fill(recoETT);
  h_efficiencyHTT_total_->Fill(recoHTT);

  if (l1MET > 40) { // TODO: add 60, 80, 100, 120
    h_efficiencyMET_pass_->Fill(recoMET);
  }

  if (l1MHT > 40) { // TODO: add 60, 80, 100, 120
    h_efficiencyMHT_pass_->Fill(recoMHT);
  }

  if (l1HTT > 120) { // TODO: add 160, 200, 240, 280
    h_efficiencyHTT_pass_->Fill(recoHTT);
  }

  if (l1ETT > 50){ // TODO: add 30, 50, 90, 140
    // TODO: check cut
    h_efficiencyETT_pass_->fill(recoETT);
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
    edm::LogError("L1TStage2CaloLayer2Offline") << "no calo jets found" << std::endl;
    return;
  }

  if (l1Jets->size() == 0) {
    edm::LogError("L1TStage2CaloLayer2Offline") << "no L1 jets found" << std::endl;
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
    edm::LogError("L1TStage2CaloLayer2Offline") << "Could not find a matching L1 jet " << std::endl;
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
  double resolutionEta = abs(recoEta) > 0 ? (l1Eta - recoEta) / recoEta : outOfBounds;
  double resolutionPhi = abs(recoPhi) > 0 ? (l1Phi - recoPhi) / recoPhi : outOfBounds;

  double minEtForTurnOns = 36.;

  // eta
  h_L1JetEtavsCaloJetEta_->Fill(recoEta, l1Eta);
  h_resolutionJetEta_->Fill(resolutionEta);

  if (abs(recoEta) <= 1.479) { // barrel
    // et
    h_L1JetETvsCaloJetET_HB_->Fill(recoEt, l1Et);
    h_L1JetETvsCaloJetET_HB_HE_->Fill(recoEt, l1Et);
    //resolution
    h_resolutionJetET_HB_->Fill(resolutionEt);
    h_resolutionJetET_HB_HE_->Fill(resolutionEt);
    // phi
    h_L1JetPhivsCaloJetPhi_HB_->Fill(recoPhi, l1Phi);
    h_L1JetPhivsCaloJetPhi_HB_HE_->Fill(recoPhi, l1Phi);
    // resolution
    h_resolutionJetPhi_HB_->Fill(resolutionPhi);
    h_resolutionJetPhi_HB_HE_->Fill(resolutionPhi);

    // turn-ons
    h_efficiencyJetEt_HB_total_->Fill(recoEt);
    h_efficiencyJetEt_HB_HE_total_->Fill(recoEt);
    if (l1Et > minEtForTurnOns) {
      h_efficiencyJetEt_HB_pass_->Fill(recoEt);
      h_efficiencyJetEt_HB_HE_pass_->Fill(recoEt);
    }

  } else if (abs(recoEta) <= 3.0) { // end-cap
    // et
    h_L1JetETvsCaloJetET_HE_->Fill(recoEt, l1Et);
    h_L1JetETvsCaloJetET_HB_HE_->Fill(recoEt, l1Et);
    //resolution
    h_resolutionJetET_HE_->Fill(resolutionEt);
    h_resolutionJetET_HB_HE_->Fill(resolutionEt);
    // phi
    h_L1JetPhivsCaloJetPhi_HE_->Fill(recoPhi, l1Phi);
    h_L1JetPhivsCaloJetPhi_HB_HE_->Fill(recoPhi, l1Phi);
    // resolution
    h_resolutionJetPhi_HE_->Fill(resolutionPhi);
    h_resolutionJetPhi_HB_HE_->Fill(resolutionPhi);

    // turn-ons
    h_efficiencyJetEt_HE_total_->Fill(recoEt);
    h_efficiencyJetEt_HB_HE_total_->Fill(recoEt);
    if (l1Et > minEtForTurnOns) {
      h_efficiencyJetEt_HE_pass_->Fill(recoEt);
      h_efficiencyJetEt_HB_HE_pass_->Fill(recoEt);
    }
  } else { // forward jets
    // et
    h_L1JetETvsCaloJetET_HF_->Fill(recoEt, l1Et);
    // resolution
    h_resolutionJetET_HF_->Fill(resolutionEt);
    // phi
    h_L1JetPhivsCaloJetPhi_HF_->Fill(recoPhi, l1Phi);
    // resolution
    h_resolutionJetPhi_HF_->Fill(resolutionPhi);
    // turn-ons
    h_efficiencyJetEt_HF_total_->Fill(recoEt);
    if (l1Et > minEtForTurnOns) {
      h_efficiencyJetEt_HF_pass_->Fill(recoEt);
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
  h_L1METvsCaloMET_ = ibooker.book2D("L1METvsCaloMET", "L1 MET vs calo MET;calo MET;L1 MET", 500, -0.5, 499.5, 500,
      -0.5, 499.5);
  h_L1MHTvsRecoMHT_ = ibooker.book2D("L1MHTvsRecoMHT", "L1 MHT vs reco MHT;reco MHT;L1 MHT", 500, -0.5, 499.5, 500,
      -0.5, 499.5);
  h_L1METTvsCaloETT_ = ibooker.book2D("L1ETTvsCaloETT", "L1 ETT vs calo ETT;calo ETT;L1 ETT", 500, -0.5, 499.5, 500,
      -0.5, 499.5);
  h_L1HTTvsRecoHTT_ = ibooker.book2D("L1HTTvsRecoHTT", "L1 HTT vs reco HTT;reco HTT;L1 HTT", 500, -0.5, 499.5, 500,
      -0.5, 499.5);

  h_L1METPhivsCaloMETPhi_ = ibooker.book2D("L1METPhivsCaloMETPhi",
      "L1 MET #phi vs calo MET #phi;calo MET #phi;L1 MET #phi", 100, -4, 4, 100, -4, 4);
  h_L1MHTPhivsRecoMHTPhi_ = ibooker.book2D("L1MHTPhivsRecoMHTPhi",
      "L1 MHT #phi vs reco MHT #phi;reco MHT #phi;L1 MHT #phi", 100, -4, 4, 100, -4, 4);

  // energy sum resolutions
  h_resolutionMET_ = ibooker.book1D("resolutionMET", "MET resolution; (L1 MET - calo MET)/calo MET; events", 50, -1,
      1.5);
  h_resolutionMHT_ = ibooker.book1D("resolutionMHT", "MHT resolution; (L1 MHT - reco MHT)/reco MHT; events", 50, -1,
      1.5);
  h_resolutionETT_ = ibooker.book1D("resolutionETT", "ETT resolution; (L1 ETT - calo ETT)/calo ETT; events", 50, -1,
      1.5);
  h_resolutionHTT_ = ibooker.book1D("resolutionHTT", "HTT resolution; (L1 HTT - reco HTT)/reco HTT; events", 50, -1,
      1.5);

  h_resolutionMETPhi_ = ibooker.book1D("resolutionMETPhi",
      "MET #phi resolution; (L1 MET #phi - reco MET #phi)/reco MET #phi; events", 120, -0.3, 0.3);
  h_resolutionMHTPhi_ = ibooker.book1D("resolutionMHTPhi",
      "MET #phi resolution; (L1 MHT #phi - reco MHT #phi)/reco MHT #phi; events", 120, -0.3, 0.3);

  // energy sum turn ons
  h_efficiencyMET_pass_ = ibooker.book1D("efficiencyMET_Num", "MET efficiency; calo MET; events", 50, -0.5, 499.5);
  h_efficiencyMHT_pass_ = ibooker.book1D("efficiencyMHT_Num", "MHT efficiency; reco MHT; events", 50, -0.5, 499.5);
  h_efficiencyETT_pass_ = ibooker.book1D("efficiencyETT_Num", "ETT efficiency; calo ETT; events", 50, -0.5, 499.5);
  h_efficiencyHTT_pass_ = ibooker.book1D("efficiencyHTT_Num", "HTT efficiency; reco HTT; events", 50, -0.5, 499.5);

  h_efficiencyMET_total_ = ibooker.book1D("efficiencyMET_Den", "MET efficiency; calo MET; events", 50, -0.5, 499.5);
  h_efficiencyMHT_total_ = ibooker.book1D("efficiencyMHT_Den", "MHT efficiency; reco MHT; events", 50, -0.5, 499.5);
  h_efficiencyETT_total_ = ibooker.book1D("efficiencyETT_Den", "ETT efficiency; calo ETT; events", 50, -0.5, 499.5);
  h_efficiencyHTT_total_ = ibooker.book1D("efficiencyHTT_Den", "HTT efficiency; reco HTT; events", 50, -0.5, 499.5);

  ibooker.cd();
}

void L1TStage2CaloLayer2Offline::bookJetHistos(DQMStore::IBooker & ibooker)
{
  ibooker.cd();
  ibooker.setCurrentFolder(histFolder_.c_str());
  // jet reco vs L1
  h_L1JetETvsCaloJetET_HB_ = ibooker.book2D("L1JetETvsCaloJetET_HB",
      "L1 jet ET vs calo jet ET (HB); calo jet ET; L1 jet ET", 300, 0, 300, 300, 0, 300);
  h_L1JetETvsCaloJetET_HE_ = ibooker.book2D("L1JetETvsCaloJetET_HE",
      "L1 jet ET vs calo jet ET (HE); calo jet ET; L1 jet ET", 300, 0, 300, 300, 0, 300);
  h_L1JetETvsCaloJetET_HF_ = ibooker.book2D("L1JetETvsCaloJetET_HF",
      "L1 jet ET vs calo jet ET (HF); calo jet ET; L1 jet ET", 300, 0, 300, 300, 0, 300);
  h_L1JetETvsCaloJetET_HB_HE_ = ibooker.book2D("L1JetETvsCaloJetET_HB_HE",
      "L1 jet ET vs calo jet ET (HB+HE); calo jet ET; L1 jet ET", 300, 0, 300, 300, 0, 300);

  h_L1JetPhivsCaloJetPhi_HB_ = ibooker.book2D("L1JetETvsCaloJetET_HB",
      "L1 jet #phi vs calo jet #phi (HB); calo jet #phi; L1 jet #phi", 100, -4, 4, 100, -4, 4);
  h_L1JetPhivsCaloJetPhi_HE_ = ibooker.book2D("L1JetETvsCaloJetET_HE",
      "L1 jet #phi vs calo jet #phi (HE); calo jet #phi; L1 jet #phi", 100, -4, 4, 100, -4, 4);
  h_L1JetPhivsCaloJetPhi_HF_ = ibooker.book2D("L1JetETvsCaloJetET_HF",
      "L1 jet #phi vs calo jet #phi (Hf); calo jet #phi; L1 jet #phi", 100, -4, 4, 100, -4, 4);
  h_L1JetPhivsCaloJetPhi_HB_HE_ = ibooker.book2D("L1JetETvsCaloJetET_HB_HE",
      "L1 jet #phi vs calo jet #phi (HB+HE); calo jet #phi; L1 jet #phi", 100, -4, 4, 100, -4, 4);

  h_L1JetEtavsCaloJetEta_ = ibooker.book2D("L1JetEtavsCaloJetEta_HB",
      "L1 jet #eta vs calo jet #eta; calo jet #eta; L1 jet #eta", 100, -10, 10, 100, -10, 10);

  // jet resolutions
  h_resolutionJetET_HB_ = ibooker.book1D("resolutionJetET_HB",
      "jet ET resolution (HB); (L1 jet ET - calo jet ET)/calo jet ET; events", 50, -1, 1.5);
  h_resolutionJetET_HE_ = ibooker.book1D("resolutionJetET_HE",
      "jet ET resolution (HE); (L1 jet ET - calo jet ET)/calo jet ET; events", 50, -1, 1.5);
  h_resolutionJetET_HF_ = ibooker.book1D("resolutionJetET_HF",
      "jet ET resolution (HF); (L1 jet ET - calo jet ET)/calo jet ET; events", 50, -1, 1.5);
  h_resolutionJetET_HB_HE_ = ibooker.book1D("resolutionJetET_HB_HE",
      "jet ET resolution (HB+HE); (L1 jet ET - calo jet ET)/calo jet ET; events", 50, -1, 1.5);

  h_resolutionJetPhi_HB_ = ibooker.book1D("resolutionJetPhi_HB",
      "jet #phi resolution (HB); (L1 jet #phi - reco jet #phi)/reco jet #phi; events", 120, -0.3, 0.3);
  h_resolutionJetPhi_HE_ = ibooker.book1D("resolutionJetPhi_HE",
      "jet #phi resolution (HE); (L1 jet #phi - reco jet #phi)/reco jet #phi; events", 120, -0.3, 0.3);
  h_resolutionJetPhi_HF_ = ibooker.book1D("resolutionJetPhi_HF",
      "jet #phi resolution (HF); (L1 jet #phi - reco jet #phi)/reco jet #phi; events", 120, -0.3, 0.3);
  h_resolutionJetPhi_HB_HE_ = ibooker.book1D("resolutionJetPhi_HB_HE",
      "jet #phi resolution (HB+HE); (L1 jet #phi - reco jet #phi)/reco jet #phi; events", 120, -0.3, 0.3);

  h_resolutionJetEta_ = ibooker.book1D("resolutionJetEta",
      "jet #eta resolution  (HB); (L1 jet #eta - reco jet #eta)/reco jet #eta; events", 120, -0.3, 0.3);

  // jet turn-ons
  h_efficiencyJetEt_HB_pass_ = ibooker.book1D("efficiencyJetEt_HB_Num", "jet turn-on (HB); reco jet ET; events", 300, 0,
      300);
  h_efficiencyJetEt_HE_pass_ = ibooker.book1D("efficiencyJetEt_HE_Num", "jet turn-on (HE); reco jet ET; events", 300, 0,
      300);
  h_efficiencyJetEt_HF_pass_ = ibooker.book1D("efficiencyJetEt_HF_Num", "jet turn-on (HF); reco jet ET; events", 300, 0,
      300);
  h_efficiencyJetEt_HB_HE_pass_ = ibooker.book1D("efficiencyJetEt_HB_HE_Num",
      "jet turn-on (HB+HE); reco jet ET; events", 300, 0, 300);

  h_efficiencyJetEt_HB_total_ = ibooker.book1D("efficiencyJetEt_HB_Den", "jet turn-on (HB); reco jet ET; events", 300,
      0, 300);
  h_efficiencyJetEt_HE_total_ = ibooker.book1D("efficiencyJetEt_HE_Den", "jet turn-on (HE); reco jet ET; events", 300,
      0, 300);
  h_efficiencyJetEt_HF_total_ = ibooker.book1D("efficiencyJetEt_HF_Den", "jet turn-on (HF); reco jet ET; events", 300,
      0, 300);
  h_efficiencyJetEt_HB_HE_total_ = ibooker.book1D("efficiencyJetEt_HB_HE_Den",
      "jet turn-on (HB+HE); reco jet ET; events", 300, 0, 300);

  ibooker.cd();
}

//
// -------------------------------------- functions --------------------------------------------
//
double L1TStage2CaloLayer2Offline::Distance(const reco::Candidate & c1, const reco::Candidate & c2)
{
  return deltaR(c1, c2);
}

double L1TStage2CaloLayer2Offline::DistancePhi(const reco::Candidate & c1, const reco::Candidate & c2)
{
  return deltaPhi(c1.p4().phi(), c2.p4().phi());
}

// This always returns only a positive deltaPhi
double L1TStage2CaloLayer2Offline::calcDeltaPhi(double phi1, double phi2)
{
  double deltaPhi = phi1 - phi2;
  if (deltaPhi < 0)
    deltaPhi = -deltaPhi;
  if (deltaPhi > 3.1415926) {
    deltaPhi = 2 * 3.1415926 - deltaPhi;
  }
  return deltaPhi;
}

//define this as a plug-in
DEFINE_FWK_MODULE (L1TStage2CaloLayer2Offline);
