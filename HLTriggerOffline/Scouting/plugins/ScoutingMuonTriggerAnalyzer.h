/*
Class declaration for ScoutingMuonTriggerAnalyzer.cc. Declares each
histogram (MonitorElement), and any functions used in
ScoutingMuonTriggerAnalyzer.cc. Also declares the token to read the
scouting muon collection and to access trigers selected in
(selected in python/ScoutingMuonTriggerAnalyzer_cfi.py)

Author: Javier Garcia de Castro, email:javigdc@bu.edu
*/

// Files to include
#ifndef DQMOffline_Scouting_ScoutingMuonTriggerAnalyzer_h
#define DQMOffline_Scouting_ScoutingMuonTriggerAnalyzer_h
#include <string>
#include <vector>

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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionParser.h"
#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"

// Classes to be declared
class ScoutingMuonTriggerAnalyzer : public DQMEDAnalyzer {
 public:
  explicit ScoutingMuonTriggerAnalyzer(const edm::ParameterSet& conf);
  ~ScoutingMuonTriggerAnalyzer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&,
                      edm::EventSetup const&) override;
  std::string outputInternalPath_;
  edm::EDGetTokenT<std::vector<pat::Muon>> muonCollection_;
  edm::EDGetTokenT<std::vector<Run3ScoutingMuon>> scoutingMuonCollection_;
  triggerExpression::Data triggerCache_;
  std::vector<triggerExpression::Evaluator*> vtriggerSelector_;
  std::vector<std::string> vtriggerSelection_;
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

#endif
