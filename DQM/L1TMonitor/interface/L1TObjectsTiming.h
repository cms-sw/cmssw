#ifndef DQM_L1TMonitor_L1TObjectsTiming_h
#define DQM_L1TMonitor_L1TObjectsTiming_h

// System include files
#include <memory>
#include <vector>
#include <array>
#include <algorithm>
#include <string>

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

class L1TObjectsTiming : public DQMEDAnalyzer {
public:
  L1TObjectsTiming(const edm::ParameterSet& ps);
  ~L1TObjectsTiming() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<l1t::MuonBxCollection> ugmtMuonToken_;
  edm::EDGetTokenT<l1t::JetBxCollection> stage2CaloLayer2JetToken_;
  edm::EDGetTokenT<l1t::EGammaBxCollection> stage2CaloLayer2EGammaToken_;
  edm::EDGetTokenT<l1t::TauBxCollection> stage2CaloLayer2TauToken_;
  edm::EDGetTokenT<l1t::EtSumBxCollection> stage2CaloLayer2EtSumToken_;

  edm::EDGetTokenT<GlobalAlgBlkBxCollection> l1tStage2uGtProducer_;  // input tag for L1 uGT DAQ readout record

  std::string monitorDir_;
  bool verbose_;

  // To get the algo bits corresponding to algo names
  std::shared_ptr<l1t::L1TGlobalUtil> gtUtil_;

  // For the timing histograms
  int algoBitFirstBxInTrain_;
  int algoBitLastBxInTrain_;
  int algoBitIsoBx_;
  const std::string algoNameFirstBxInTrain_;
  const std::string algoNameLastBxInTrain_;
  const std::string algoNameIsoBx_;
  const unsigned int bxrange_;  //this is the out bx range

  unsigned int useAlgoDecision_;

  std::vector<double> egammaPtCuts_;
  double jetPtCut_;
  double egammaPtCut_;
  double tauPtCut_;
  double etsumPtCut_;
  double muonPtCut_;
  int muonQualCut_;

  //---------Histograms booking---------
  // All bunches
  std::vector<MonitorElement*> muons_eta_phi;
  std::vector<MonitorElement*> jet_eta_phi;
  std::vector<MonitorElement*> egamma_eta_phi;
  std::vector<MonitorElement*> tau_eta_phi;
  std::vector<MonitorElement*> etsum_eta_phi_MET;
  std::vector<MonitorElement*> etsum_eta_phi_METHF;
  std::vector<MonitorElement*> etsum_eta_phi_MHT;
  std::vector<MonitorElement*> etsum_eta_phi_MHTHF;

  MonitorElement* denominator_muons;
  MonitorElement* denominator_jet;
  MonitorElement* denominator_egamma;
  MonitorElement* denominator_tau;
  MonitorElement* denominator_etsum_MET;
  MonitorElement* denominator_etsum_METHF;
  MonitorElement* denominator_etsum_MHT;
  MonitorElement* denominator_etsum_MHTHF;

  // Isolated bunches
  std::vector<MonitorElement*> muons_eta_phi_isolated;
  std::vector<MonitorElement*> jet_eta_phi_isolated;
  std::vector<std::vector<MonitorElement*>> egamma_eta_phi_isolated;
  std::vector<MonitorElement*> tau_eta_phi_isolated;
  std::vector<MonitorElement*> etsum_eta_phi_MET_isolated;
  std::vector<MonitorElement*> etsum_eta_phi_METHF_isolated;
  std::vector<MonitorElement*> etsum_eta_phi_MHT_isolated;
  std::vector<MonitorElement*> etsum_eta_phi_MHTHF_isolated;

  MonitorElement* denominator_muons_isolated;
  MonitorElement* denominator_jet_isolated;
  std::vector<MonitorElement*> denominator_egamma_isolated;
  MonitorElement* denominator_tau_isolated;
  MonitorElement* denominator_etsum_isolated_MET;
  MonitorElement* denominator_etsum_isolated_METHF;
  MonitorElement* denominator_etsum_isolated_MHT;
  MonitorElement* denominator_etsum_isolated_MHTHF;

  std::vector<MonitorElement*> egamma_iso_bx_ieta_isolated;
  std::vector<MonitorElement*> egamma_noniso_bx_ieta_isolated;

  // First bunch in train
  std::vector<MonitorElement*> muons_eta_phi_firstbunch;
  std::vector<MonitorElement*> jet_eta_phi_firstbunch;
  std::vector<std::vector<MonitorElement*>> egamma_eta_phi_firstbunch;
  std::vector<MonitorElement*> tau_eta_phi_firstbunch;
  std::vector<MonitorElement*> etsum_eta_phi_firstbunch;
  std::vector<MonitorElement*> etsum_eta_phi_MET_firstbunch;
  std::vector<MonitorElement*> etsum_eta_phi_METHF_firstbunch;
  std::vector<MonitorElement*> etsum_eta_phi_MHT_firstbunch;
  std::vector<MonitorElement*> etsum_eta_phi_MHTHF_firstbunch;

  MonitorElement* denominator_muons_firstbunch;
  MonitorElement* denominator_jet_firstbunch;
  std::vector<MonitorElement*> denominator_egamma_firstbunch;
  MonitorElement* denominator_tau_firstbunch;
  MonitorElement* denominator_etsum_firstbunch_MET;
  MonitorElement* denominator_etsum_firstbunch_METHF;
  MonitorElement* denominator_etsum_firstbunch_MHT;
  MonitorElement* denominator_etsum_firstbunch_MHTHF;

  std::vector<MonitorElement*> egamma_iso_bx_ieta_firstbunch;
  std::vector<MonitorElement*> egamma_noniso_bx_ieta_firstbunch;

  // Last bunch in train
  std::vector<MonitorElement*> muons_eta_phi_lastbunch;
  std::vector<MonitorElement*> jet_eta_phi_lastbunch;
  std::vector<std::vector<MonitorElement*>> egamma_eta_phi_lastbunch;
  std::vector<MonitorElement*> tau_eta_phi_lastbunch;
  std::vector<MonitorElement*> etsum_eta_phi_MET_lastbunch;
  std::vector<MonitorElement*> etsum_eta_phi_METHF_lastbunch;
  std::vector<MonitorElement*> etsum_eta_phi_MHT_lastbunch;
  std::vector<MonitorElement*> etsum_eta_phi_MHTHF_lastbunch;

  MonitorElement* denominator_muons_lastbunch;
  MonitorElement* denominator_jet_lastbunch;
  std::vector<MonitorElement*> denominator_egamma_lastbunch;
  MonitorElement* denominator_tau_lastbunch;
  MonitorElement* denominator_etsum_lastbunch_MET;
  MonitorElement* denominator_etsum_lastbunch_METHF;
  MonitorElement* denominator_etsum_lastbunch_MHT;
  MonitorElement* denominator_etsum_lastbunch_MHTHF;

  std::vector<MonitorElement*> egamma_iso_bx_ieta_lastbunch;
  std::vector<MonitorElement*> egamma_noniso_bx_ieta_lastbunch;
};

#endif
