/*
  Scouting Muon DQM for L1 seeds. 
  This code does the following:

     1) Reads pat muon and scouting muon collections, and writes an array of
scouting muon triggers (selected in python/ScoutingMuonTriggerAnalyzer_cfi.py)

     2) For each event, if the event passes a logical OR of HLTriggers it is added
     to the denominator, and if it passes any of the scouting muon triggers it is
     added to the numerator of that specific trigger.

     3) Fills histograms for both leading and subleading muon in the event.
  
  Author: Javier Garcia de Castro, email:javigdc@bu.edu
*/

// system includes
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// user includes
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionParser.h"
#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"

// Classes to be declared
class ScoutingMuonTriggerAnalyzer : public DQMEDAnalyzer {
public:
  explicit ScoutingMuonTriggerAnalyzer(const edm::ParameterSet& conf);
  ~ScoutingMuonTriggerAnalyzer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  // data members
  const std::string outputInternalPath_;
  triggerExpression::Data triggerCache_;
  const std::vector<std::string> vtriggerSelection_;

  const edm::EDGetTokenT<std::vector<pat::Muon>> muonCollection_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingMuon>> scoutingMuonCollection_;

  std::vector<triggerExpression::Evaluator*> vtriggerSelector_;
  edm::EDGetToken algToken_;
  std::shared_ptr<l1t::L1TGlobalUtil> l1GtUtils_;
  std::vector<std::string> l1Seeds_;
  TString l1Names[100] = {""};
  Bool_t l1Result[100] = {false};

  // Histogram declaration
  // DENOMINATORS:
  dqm::reco::MonitorElement* h_invMass_denominator;
  dqm::reco::MonitorElement* h_pt1_l1_denominator;
  dqm::reco::MonitorElement* h_eta1_l1_denominator;
  dqm::reco::MonitorElement* h_phi1_l1_denominator;
  dqm::reco::MonitorElement* h_dxy1_l1_denominator;
  dqm::reco::MonitorElement* h_pt2_l1_denominator;
  dqm::reco::MonitorElement* h_eta2_l1_denominator;
  dqm::reco::MonitorElement* h_phi2_l1_denominator;
  dqm::reco::MonitorElement* h_dxy2_l1_denominator;

  // NUMERATORS:
  std::vector<dqm::reco::MonitorElement*> h_invMass_numerators;
  std::vector<dqm::reco::MonitorElement*> h_pt1_l1_numerators;
  std::vector<dqm::reco::MonitorElement*> h_eta1_l1_numerators;
  std::vector<dqm::reco::MonitorElement*> h_phi1_l1_numerators;
  std::vector<dqm::reco::MonitorElement*> h_dxy1_l1_numerators;
  std::vector<dqm::reco::MonitorElement*> h_pt2_l1_numerators;
  std::vector<dqm::reco::MonitorElement*> h_eta2_l1_numerators;
  std::vector<dqm::reco::MonitorElement*> h_phi2_l1_numerators;
  std::vector<dqm::reco::MonitorElement*> h_dxy2_l1_numerators;
};

// Read the collections and triggers
ScoutingMuonTriggerAnalyzer::ScoutingMuonTriggerAnalyzer(const edm::ParameterSet& iConfig)
    : outputInternalPath_{iConfig.getParameter<std::string>("OutputInternalPath")},
      triggerCache_{triggerExpression::Data(iConfig.getParameterSet("triggerConfiguration"), consumesCollector())},
      vtriggerSelection_{iConfig.getParameter<vector<string>>("triggerSelection")},
      scoutingMuonCollection_{
          consumes<std::vector<Run3ScoutingMuon>>(iConfig.getParameter<edm::InputTag>("ScoutingMuonCollection"))},
      algToken_{consumes<BXVector<GlobalAlgBlk>>(iConfig.getParameter<edm::InputTag>("AlgInputTag"))} {
  vtriggerSelector_.reserve(vtriggerSelection_.size());
  for (auto const& vt : vtriggerSelection_)
    vtriggerSelector_.push_back(triggerExpression::parse(vt));
  l1GtUtils_ = std::make_shared<l1t::L1TGlobalUtil>(iConfig, consumesCollector(), l1t::UseEventSetupIn::RunAndEvent);
  l1Seeds_ = iConfig.getParameter<std::vector<std::string>>("l1Seeds");
  for (unsigned int i = 0; i < l1Seeds_.size(); i++) {
    const auto& l1seed(l1Seeds_.at(i));
    l1Names[i] = TString(l1seed);
  }
}

// Core of the implementation
void ScoutingMuonTriggerAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  edm::Handle<std::vector<Run3ScoutingMuon>> sctMuons;
  iEvent.getByToken(scoutingMuonCollection_, sctMuons);
  if (sctMuons.failedToGet()) {
    edm::LogWarning("ScoutingMonitoring") << "Run3ScoutingMuon collection not found.";
    return;
  }

  // Check whether events pass any of the HLTriggers to add to the denominator
  bool passHLTDenominator = false;
  if (triggerCache_.setEvent(iEvent, iSetup)) {
    for (unsigned int i = 0; i < vtriggerSelector_.size(); i++) {
      auto& vts(vtriggerSelector_.at(i));
      bool result = false;
      if (vts) {
        if (triggerCache_.configurationUpdated())
          vts->init(triggerCache_);
        result = (*vts)(triggerCache_);
      }
      if (result)
        passHLTDenominator = true;
    }
  }

  // Find leading and subleading muon from the event
  if (!sctMuons->empty()) {
    Run3ScoutingMuon leading_mu;
    Run3ScoutingMuon subleading_mu;

    std::vector<Run3ScoutingMuon> sorted_mu;
    for (const auto& muon : *sctMuons) {
      sorted_mu.push_back(muon);
    }
    std::sort(std::begin(sorted_mu), std::end(sorted_mu), [&](Run3ScoutingMuon mu1, Run3ScoutingMuon mu2) {
      return mu1.pt() > mu2.pt();
    });
    leading_mu = sorted_mu.at(0);
    if (sorted_mu.size() > 1)
      subleading_mu = sorted_mu.at(1);

    l1GtUtils_->retrieveL1(iEvent, iSetup, algToken_);

    math::PtEtaPhiMLorentzVector mu1(leading_mu.pt(), leading_mu.eta(), leading_mu.phi(), leading_mu.m());
    math::PtEtaPhiMLorentzVector mu2(subleading_mu.pt(), subleading_mu.eta(), subleading_mu.phi(), subleading_mu.m());
    float invMass = (mu1 + mu2).mass();
    // If event passed and of the HLTs, add to denominator
    if (passHLTDenominator) {
      h_invMass_denominator->Fill(invMass);
      h_pt1_l1_denominator->Fill(leading_mu.pt());
      h_eta1_l1_denominator->Fill(leading_mu.eta());
      h_phi1_l1_denominator->Fill(leading_mu.phi());
      h_dxy1_l1_denominator->Fill(leading_mu.trk_dxy());
      if (sorted_mu.size() > 1) {
        h_pt2_l1_denominator->Fill(subleading_mu.pt());
        h_eta2_l1_denominator->Fill(subleading_mu.eta());
        h_phi2_l1_denominator->Fill(subleading_mu.phi());
        h_dxy2_l1_denominator->Fill(subleading_mu.trk_dxy());
      }

      // For each L1 seed, if the event passes the trigger plot distributions in
      // the numerator
      for (unsigned int i = 0; i < l1Seeds_.size(); i++) {
        const auto& l1seed(l1Seeds_.at(i));
        bool l1htbit = false;
        double prescale = -1;
        l1GtUtils_->getFinalDecisionByName(l1seed, l1htbit);
        l1GtUtils_->getPrescaleByName(l1seed, prescale);
        l1Result[i] = l1htbit;
        if (l1Result[i] == 1) {
          h_invMass_numerators[i]->Fill(invMass);
          h_pt1_l1_numerators[i]->Fill(leading_mu.pt());
          h_eta1_l1_numerators[i]->Fill(leading_mu.eta());
          h_phi1_l1_numerators[i]->Fill(leading_mu.phi());
          h_dxy1_l1_numerators[i]->Fill(leading_mu.trk_dxy());
          if (sorted_mu.size() > 1) {
            h_pt2_l1_numerators[i]->Fill(subleading_mu.pt());
            h_eta2_l1_numerators[i]->Fill(subleading_mu.eta());
            h_phi2_l1_numerators[i]->Fill(subleading_mu.phi());
            h_dxy2_l1_numerators[i]->Fill(subleading_mu.trk_dxy());
          }
        }
      }
    }
  }
}

// Histogram axes labels, bin number and range
void ScoutingMuonTriggerAnalyzer::bookHistograms(DQMStore::IBooker& ibook,
                                                 edm::Run const& run,
                                                 edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(outputInternalPath_);
  h_invMass_denominator = ibook.book1D("h_invMass_denominator", ";Invariant Mass (GeV); Muons", 100, 0.0, 20.0);
  h_pt1_l1_denominator = ibook.book1D("h_pt1_denominator", ";Leading muon pt (GeV); Muons", 100, 0, 50.0);
  h_eta1_l1_denominator = ibook.book1D("h_eta1_denominator", ";Leading muon eta; Muons", 100, -5.0, 5.0);
  h_phi1_l1_denominator = ibook.book1D("h_phi1_denominator", ";Leading muon phi; Muons", 100, -3.3, 3.3);
  h_dxy1_l1_denominator = ibook.book1D("h_dxy1_denominator", ";Leading muon dxy; Muons", 100, 0, 5.0);
  h_pt2_l1_denominator = ibook.book1D("h_pt2_denominator", ";Subleading muon pt (GeV); Muons", 100, 0, 50.0);
  h_eta2_l1_denominator = ibook.book1D("h_eta2_denominator", ";Subleading muon eta; Muons", 100, -5.0, 5.0);
  h_phi2_l1_denominator = ibook.book1D("h_phi2_denominator", ";Subleading muon phi; Muons", 100, -3.3, 3.3);
  h_dxy2_l1_denominator = ibook.book1D("h_dxy2_denominator", ";Subleaing muon dxy; Muons", 100, 0, 5.0);

  for (unsigned int i = 0; i < l1Seeds_.size(); i++) {
    const auto& l1seed = l1Seeds_.at(i);
    h_invMass_numerators.push_back(
        ibook.book1D(Form("h_invMass_numerator_%s", l1seed.c_str()), ";Invariant mass (GeV); Muons", 100, 0.0, 20.0));
    h_pt1_l1_numerators.push_back(
        ibook.book1D(Form("h_pt1_numerator_%s", l1seed.c_str()), ";Leading muon pt (GeV); Muons", 100, 0, 50.0));
    h_eta1_l1_numerators.push_back(
        ibook.book1D(Form("h_eta1_numerator_%s", l1seed.c_str()), ";Leading muon eta; Muons", 100, -5.0, 5.0));
    h_phi1_l1_numerators.push_back(
        ibook.book1D(Form("h_phi1_numerator_%s", l1seed.c_str()), ";Leading muon phi; Muons", 100, 3.3, -3.3));
    h_dxy1_l1_numerators.push_back(
        ibook.book1D(Form("h_dxy1_numerator_%s", l1seed.c_str()), ";Leading muon dxy; Muons", 100, 0, 5.0));
    h_pt2_l1_numerators.push_back(
        ibook.book1D(Form("h_pt2_numerator_%s", l1seed.c_str()), ";Subleading muon pt (GeV); Muons", 100, 0, 50.0));
    h_eta2_l1_numerators.push_back(
        ibook.book1D(Form("h_eta2_numerator_%s", l1seed.c_str()), ";Subleading muon eta; Muons", 100, -5.0, 5.0));
    h_phi2_l1_numerators.push_back(
        ibook.book1D(Form("h_phi2_numerator_%s", l1seed.c_str()), ";Subleading muon phi; Muons", 100, 3.3, -3.3));
    h_dxy2_l1_numerators.push_back(
        ibook.book1D(Form("h_dxy2_numerator_%s", l1seed.c_str()), ";Subleading muon dxy; Muons", 100, 0, 5.0));
  }
}

// Input tags to read collections and L1 seeds
void ScoutingMuonTriggerAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("OutputInternalPath", "MY_FOLDER");
  desc.add<vector<string>>("triggerSelection", {});
  desc.add<edm::InputTag>("ScoutingMuonCollection", edm::InputTag("hltScoutingMuonPackerVtx"));
  desc.add<edm::InputTag>("AlgInputTag", edm::InputTag("gtStage2Digis"));
  desc.add<std::vector<std::string>>("l1Seeds", {});
  desc.add<edm::InputTag>("l1tAlgBlkInputTag", edm::InputTag("gtStage2Digis"));
  desc.add<edm::InputTag>("l1tExtBlkInputTag", edm::InputTag("gtStage2Digis"));
  desc.add<bool>("ReadPrescalesFromFile", false);
  edm::ParameterSetDescription triggerConfig;
  triggerConfig.setAllowAnything();
  desc.add<edm::ParameterSetDescription>("triggerConfiguration", triggerConfig);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(ScoutingMuonTriggerAnalyzer);
