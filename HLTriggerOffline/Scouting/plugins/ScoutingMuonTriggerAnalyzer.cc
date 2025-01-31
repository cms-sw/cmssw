#include "ScoutingMuonTriggerAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <cmath>

ScoutingMuonTriggerAnalyzer::ScoutingMuonTriggerAnalyzer(const edm::ParameterSet& iConfig)
    : outputInternalPath_(iConfig.getParameter<std::string>("OutputInternalPath")),
      triggerCache_(triggerExpression::Data(iConfig.getParameterSet("triggerConfiguration"), consumesCollector())),
      vtriggerSelection_(iConfig.getParameter<vector<string>>("triggerSelection")) {
  //outputInternalPath_ = iConfig.getParameter<std::string>("OutputInternalPath");
  muonCollection_ = consumes<std::vector<pat::Muon>>(iConfig.getParameter<edm::InputTag>("MuonCollection"));
  scoutingMuonCollection_ =
      consumes<std::vector<Run3ScoutingMuon>>(iConfig.getParameter<edm::InputTag>("ScoutingMuonCollection"));
  vtriggerSelector_.reserve(vtriggerSelection_.size());
  for (auto const& vt : vtriggerSelection_)
    vtriggerSelector_.push_back(triggerExpression::parse(vt));
  algToken_ = consumes<BXVector<GlobalAlgBlk>>(iConfig.getParameter<edm::InputTag>("AlgInputTag"));
  l1GtUtils_ = std::make_shared<l1t::L1TGlobalUtil>(iConfig, consumesCollector(), l1t::UseEventSetupIn::RunAndEvent);
  l1Seeds_ = iConfig.getParameter<std::vector<std::string>>("l1Seeds");
  for (unsigned int i = 0; i < l1Seeds_.size(); i++) {
    const auto& l1seed(l1Seeds_.at(i));
    l1Names[i] = TString(l1seed);
  }
}

ScoutingMuonTriggerAnalyzer::~ScoutingMuonTriggerAnalyzer() {}

void ScoutingMuonTriggerAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  edm::Handle<std::vector<Run3ScoutingMuon>> sctMuons;
  iEvent.getByToken(scoutingMuonCollection_, sctMuons);
  if (sctMuons.failedToGet()) {
    edm::LogWarning("ScoutingMonitoring") << "Run3ScoutingMuon collection not found.";
    return;
  }

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
    //if((2.4 < invMass && invMass < 3.8) || (3.5 < invMass && invMass < 3.9) || (85.2 < invMass && invMass < 97.2) || (8.5 < invMass && invMass < 10.5)){
    if (passHLTDenominator) {
      /*
        if(2.4 < invMass && invMass < 3.8){
          h_invMass_denominator_JPsi->Fill(invMass);
        }
        if(3.5 < invMass && invMass <3.9 ){
          h_invMass_denominator_Psi2->Fill(invMass);
        }
        if(8.5 < invMass && invMass < 10.5){
          h_invMass_denominator_Upsilon->Fill(invMass);
        }
        if(85.2 < invMass && invMass < 97.2){
          h_invMass_denominator_Z->Fill(invMass);
        }
        */
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

      for (unsigned int i = 0; i < l1Seeds_.size(); i++) {
        const auto& l1seed(l1Seeds_.at(i));
        bool l1htbit = false;
        double prescale = -1;
        l1GtUtils_->getFinalDecisionByName(l1seed, l1htbit);
        l1GtUtils_->getPrescaleByName(l1seed, prescale);
        l1Result[i] = l1htbit;
        if (l1Result[i] == 1) {
          /*
            if(2.4 < invMass && invMass < 3.8){
              h_invMass_numerators_JPsi[i]->Fill(invMass);
            }
            if(3.5 < invMass && invMass <3.9 ){
              h_invMass_numerators_Psi2[i]->Fill(invMass);
            }
            if(8.5 < invMass && invMass < 10.5){
              h_invMass_numerators_Upsilon[i]->Fill(invMass);
            }
            if(85.2 < invMass && invMass < 97.2){
              h_invMass_numerators_Z[i]->Fill(invMass);
            }
            */
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
    //}
  }
}
void ScoutingMuonTriggerAnalyzer::bookHistograms(DQMStore::IBooker& ibook,
                                                 edm::Run const& run,
                                                 edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(outputInternalPath_);
  h_invMass_denominator = ibook.book1D("h_invMass_denominator", ";Invariant Mass (GeV); Muons", 100, 0.0, 20.0);
  //h_invMass_denominator_JPsi = ibook.book1D("h_invMass_denominator", ";Invariant Mass (GeV); Muons", 100, 2.3, 3.9);
  //h_invMass_denominator_Psi2 = ibook.book1D("h_invMass_denominator", ";Invariant Mass (GeV); Muons", 100, 3.4, 4.0);
  //h_invMass_denominator_Upsilon = ibook.book1D("h_invMass_denominator", ";Invariant Mass (GeV); Muons", 100, 8.4, 10.4);
  //h_invMass_denominator_Z = ibook.book1D("h_invMass_denominator", ";Invariant Mass (GeV); Muons", 100, 84, 98);
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
    //h_invMass_numerators_JPsi.push_back(ibook.book1D(Form("h_invMass_numerator_JPsi_%s", l1seed.c_str()), ";Invariant mass (GeV); Muons", 100, 2.3, 3.9));
    //h_invMass_numerators_Psi2.push_back(ibook.book1D(Form("h_invMass_numerator_Psi2_%s", l1seed.c_str()), ";Invariant mass (GeV); Muons", 100, 3.4, 4.0));
    //h_invMass_numerators_Upsilon.push_back(ibook.book1D(Form("h_invMass_numerator_Upsilon_%s", l1seed.c_str()), ";Invariant mass (GeV); Muons", 100, 8.4, 10.4));
    //h_invMass_numerators_Z.push_back(ibook.book1D(Form("h_invMass_numerator_Z_%s", l1seed.c_str()), ";Invariant mass (GeV); Muons", 100, 84, 98));
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

void ScoutingMuonTriggerAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("OutputInternalPath", "MY_FOLDER");
  desc.add<edm::InputTag>("MuonCollection", edm::InputTag("slimmedMuons"));
  desc.add<edm::InputTag>("AlgInputTag", edm::InputTag("gtStage2Digis"));
  desc.add<edm::InputTag>("l1tAlgBlkInputTag", edm::InputTag("gtStage2Digis"));
  desc.add<edm::InputTag>("l1tExtBlkInputTag", edm::InputTag("gtStage2Digis"));
  desc.setUnknown();
  descriptions.addDefault(desc);
  descriptions.add("ScoutingMuonTriggerAnalyzer", desc);
}

DEFINE_FWK_MODULE(ScoutingMuonTriggerAnalyzer);