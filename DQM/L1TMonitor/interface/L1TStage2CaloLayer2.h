#ifndef DQM_L1TMonitor_L1TStage2CaloLayer2_h
#define DQM_L1TMonitor_L1TStage2CaloLayer2_h

//   base classes
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DQM
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

// stage2 collection

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

class L1TStage2CaloLayer2 : public DQMEDAnalyzer {
  
 public:
  
  L1TStage2CaloLayer2(const edm::ParameterSet & ps);

  virtual ~L1TStage2CaloLayer2();

 protected:

  void analyze(const edm::Event& e, const edm::EventSetup& c);
  virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&);
  virtual void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override ;
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);

 private:

  std::string monitorDir_;

  edm::EDGetTokenT<l1t::JetBxCollection> stage2CaloLayer2JetToken_;
  edm::EDGetTokenT<l1t::EGammaBxCollection> stage2CaloLayer2EGammaToken_;
  edm::EDGetTokenT<l1t::TauBxCollection> stage2CaloLayer2TauToken_;
  edm::EDGetTokenT<l1t::EtSumBxCollection> stage2CaloLayer2EtSumToken_;

  bool verbose_;

  MonitorElement* stage2CaloLayer2CenJetEtEtaPhi_;
  MonitorElement* stage2CaloLayer2CenJetEta_;
  MonitorElement* stage2CaloLayer2CenJetPhi_;
  MonitorElement* stage2CaloLayer2CenJetRank_;
  MonitorElement* stage2CaloLayer2CenJetOcc_;
  MonitorElement* stage2CaloLayer2CenJetBxOcc_;

  MonitorElement* stage2CaloLayer2ForJetEtEtaPhi_;
  MonitorElement* stage2CaloLayer2ForJetEta_;
  MonitorElement* stage2CaloLayer2ForJetPhi_;
  MonitorElement* stage2CaloLayer2ForJetRank_;
  MonitorElement* stage2CaloLayer2ForJetOcc_;
  MonitorElement* stage2CaloLayer2ForJetBxOcc_;

  MonitorElement* stage2CaloLayer2IsoEGEtEtaPhi_;
  MonitorElement* stage2CaloLayer2IsoEGEta_;
  MonitorElement* stage2CaloLayer2IsoEGPhi_;
  MonitorElement* stage2CaloLayer2IsoEGRank_;
  MonitorElement* stage2CaloLayer2IsoEGOcc_;
  MonitorElement* stage2CaloLayer2IsoEGBxOcc_;

  MonitorElement* stage2CaloLayer2NonIsoEGEtEtaPhi_;
  MonitorElement* stage2CaloLayer2NonIsoEGEta_;
  MonitorElement* stage2CaloLayer2NonIsoEGPhi_;
  MonitorElement* stage2CaloLayer2NonIsoEGRank_;
  MonitorElement* stage2CaloLayer2NonIsoEGOcc_;
  MonitorElement* stage2CaloLayer2NonIsoEGBxOcc_;

  MonitorElement* stage2CaloLayer2IsoTauEtEtaPhi_;
  MonitorElement* stage2CaloLayer2IsoTauEta_;
  MonitorElement* stage2CaloLayer2IsoTauPhi_;
  MonitorElement* stage2CaloLayer2IsoTauRank_;
  MonitorElement* stage2CaloLayer2IsoTauOcc_;
  MonitorElement* stage2CaloLayer2IsoTauBxOcc_;

  MonitorElement* stage2CaloLayer2TauEtEtaPhi_;
  MonitorElement* stage2CaloLayer2TauEta_;
  MonitorElement* stage2CaloLayer2TauPhi_;
  MonitorElement* stage2CaloLayer2TauRank_;
  MonitorElement* stage2CaloLayer2TauOcc_;
  MonitorElement* stage2CaloLayer2TauBxOcc_;

  MonitorElement* stage2CaloLayer2EtSumBxOcc_;
  MonitorElement* stage2CaloLayer2METRank_;
  MonitorElement* stage2CaloLayer2METPhi_;
  MonitorElement* stage2CaloLayer2ETTRank_;
  MonitorElement* stage2CaloLayer2ETTPhi_;
  MonitorElement* stage2CaloLayer2MHTRank_;
  MonitorElement* stage2CaloLayer2MHTPhi_;
  MonitorElement* stage2CaloLayer2MHTEta_;
  MonitorElement* stage2CaloLayer2HTTRank_;
  MonitorElement* stage2CaloLayer2HTTPhi_;
  MonitorElement* stage2CaloLayer2HTTEta_;
};

#endif 
