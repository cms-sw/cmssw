#ifndef DQM_L1TMonitor_L1TObjectsTiming_h
#define DQM_L1TMonitor_L1TObjectsTiming_h

// System include files
#include <memory>
#include <vector>

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
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class L1TObjectsTiming : public DQMEDAnalyzer {

 public:

  L1TObjectsTiming(const edm::ParameterSet& ps);
  ~L1TObjectsTiming() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 protected:

  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

 private:  

  edm::EDGetTokenT<l1t::MuonBxCollection> ugmtMuonToken_;
  edm::EDGetTokenT<l1t::JetBxCollection> stage2CaloLayer2JetToken_;
  edm::EDGetTokenT<l1t::EGammaBxCollection> stage2CaloLayer2EGammaToken_;
  edm::EDGetTokenT<l1t::TauBxCollection> stage2CaloLayer2TauToken_;
  edm::EDGetTokenT<l1t::EtSumBxCollection> stage2CaloLayer2EtSumToken_;

  edm::EDGetTokenT<GlobalAlgBlkBxCollection> l1tStage2uGtProducer_; // input tag for L1 uGT DAQ readout record

  std::string monitorDir_;
  bool verbose_;
  const unsigned int bxrange = 5; //this is the out bx range

  // To get the algo bits corresponding to algo names
  l1t::L1TGlobalUtil* gtUtil_;

  // For the timing histograms
  int algoBitFirstBxInTrain_;
  int algoBitLastBxInTrain_;
  const std::string algoNameFirstBxInTrain_;
  const std::string algoNameLastBxInTrain_;

  
//---------Histograms booking---------
  std::vector<MonitorElement*> muons_eta_phi;
  std::vector<MonitorElement*> jet_eta_phi;
  std::vector<MonitorElement*> egamma_eta_phi;
  std::vector<MonitorElement*> tau_eta_phi;
  std::vector<MonitorElement*> etsum_eta_phi_MET;
  std::vector<MonitorElement*> etsum_eta_phi_METHF;
  std::vector<MonitorElement*> etsum_eta_phi_MHT;
  std::vector<MonitorElement*> etsum_eta_phi_MHTHF;

  std::vector<MonitorElement*> muons_eta_phi_isolated;
  std::vector<MonitorElement*> jet_eta_phi_isolated;
  std::vector<MonitorElement*> egamma_eta_phi_isolated;
  std::vector<MonitorElement*> tau_eta_phi_isolated;
  std::vector<MonitorElement*> etsum_eta_phi_MET_isolated;
  std::vector<MonitorElement*> etsum_eta_phi_METHF_isolated;
  std::vector<MonitorElement*> etsum_eta_phi_MHT_isolated;
  std::vector<MonitorElement*> etsum_eta_phi_MHTHF_isolated;

  std::vector<MonitorElement*> muons_eta_phi_firstbunch;
  std::vector<MonitorElement*> jet_eta_phi_firstbunch;
  std::vector<MonitorElement*> egamma_eta_phi_firstbunch;
  std::vector<MonitorElement*> tau_eta_phi_firstbunch;
  std::vector<MonitorElement*> etsum_eta_phi_firstbunch;
  std::vector<MonitorElement*> etsum_eta_phi_MET_firstbunch;
  std::vector<MonitorElement*> etsum_eta_phi_METHF_firstbunch;
  std::vector<MonitorElement*> etsum_eta_phi_MHT_firstbunch;
  std::vector<MonitorElement*> etsum_eta_phi_MHTHF_firstbunch;

  std::vector<MonitorElement*> muons_eta_phi_lastbunch;
  std::vector<MonitorElement*> jet_eta_phi_lastbunch;
  std::vector<MonitorElement*> egamma_eta_phi_lastbunch;
  std::vector<MonitorElement*> tau_eta_phi_lastbunch;
  std::vector<MonitorElement*> etsum_eta_phi_MET_lastbunch;
  std::vector<MonitorElement*> etsum_eta_phi_METHF_lastbunch;
  std::vector<MonitorElement*> etsum_eta_phi_MHT_lastbunch;
  std::vector<MonitorElement*> etsum_eta_phi_MHTHF_lastbunch;

};

#endif
