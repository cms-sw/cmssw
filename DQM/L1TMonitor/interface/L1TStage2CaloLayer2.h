#ifndef DQM_L1TMonitor_L1TStage2CaloLayer2_h
#define DQM_L1TMonitor_L1TStage2CaloLayer2_h

//   base classes
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DQM
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"

// stage2 collection
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Tau.h"

namespace calolayer2dqm {
  struct Histograms {
    ConcurrentMonitorElement stage2CaloLayer2CenJetEtEtaPhi;
    ConcurrentMonitorElement stage2CaloLayer2CenJetEta;
    ConcurrentMonitorElement stage2CaloLayer2CenJetPhi;
    ConcurrentMonitorElement stage2CaloLayer2CenJetRank;
    ConcurrentMonitorElement stage2CaloLayer2CenJetOcc;
    ConcurrentMonitorElement stage2CaloLayer2CenJetBxOcc;
    ConcurrentMonitorElement stage2CaloLayer2CenJetQual;

    ConcurrentMonitorElement stage2CaloLayer2ForJetEtEtaPhi;
    ConcurrentMonitorElement stage2CaloLayer2ForJetEta;
    ConcurrentMonitorElement stage2CaloLayer2ForJetPhi;
    ConcurrentMonitorElement stage2CaloLayer2ForJetRank;
    ConcurrentMonitorElement stage2CaloLayer2ForJetOcc;
    ConcurrentMonitorElement stage2CaloLayer2ForJetBxOcc;
    ConcurrentMonitorElement stage2CaloLayer2ForJetQual;

    ConcurrentMonitorElement stage2CaloLayer2EGIso;

    ConcurrentMonitorElement stage2CaloLayer2IsoEGEtEtaPhi;
    ConcurrentMonitorElement stage2CaloLayer2IsoEGEta;
    ConcurrentMonitorElement stage2CaloLayer2IsoEGPhi;
    ConcurrentMonitorElement stage2CaloLayer2IsoEGRank;
    ConcurrentMonitorElement stage2CaloLayer2IsoEGOcc;
    ConcurrentMonitorElement stage2CaloLayer2IsoEGBxOcc;
    ConcurrentMonitorElement stage2CaloLayer2IsoEGQual;

    ConcurrentMonitorElement stage2CaloLayer2NonIsoEGEtEtaPhi;
    ConcurrentMonitorElement stage2CaloLayer2NonIsoEGEta;
    ConcurrentMonitorElement stage2CaloLayer2NonIsoEGPhi;
    ConcurrentMonitorElement stage2CaloLayer2NonIsoEGRank;
    ConcurrentMonitorElement stage2CaloLayer2NonIsoEGOcc;
    ConcurrentMonitorElement stage2CaloLayer2NonIsoEGBxOcc;
    ConcurrentMonitorElement stage2CaloLayer2NonIsoEGQual;

    ConcurrentMonitorElement stage2CaloLayer2TauIso;

    ConcurrentMonitorElement stage2CaloLayer2IsoTauEtEtaPhi;
    ConcurrentMonitorElement stage2CaloLayer2IsoTauEta;
    ConcurrentMonitorElement stage2CaloLayer2IsoTauPhi;
    ConcurrentMonitorElement stage2CaloLayer2IsoTauRank;
    ConcurrentMonitorElement stage2CaloLayer2IsoTauOcc;
    ConcurrentMonitorElement stage2CaloLayer2IsoTauBxOcc;
    ConcurrentMonitorElement stage2CaloLayer2IsoTauQual;

    ConcurrentMonitorElement stage2CaloLayer2TauEtEtaPhi;
    ConcurrentMonitorElement stage2CaloLayer2TauEta;
    ConcurrentMonitorElement stage2CaloLayer2TauPhi;
    ConcurrentMonitorElement stage2CaloLayer2TauRank;
    ConcurrentMonitorElement stage2CaloLayer2TauOcc;
    ConcurrentMonitorElement stage2CaloLayer2TauBxOcc;
    ConcurrentMonitorElement stage2CaloLayer2TauQual;

    ConcurrentMonitorElement stage2CaloLayer2EtSumBxOcc;
    ConcurrentMonitorElement stage2CaloLayer2METRank;
    ConcurrentMonitorElement stage2CaloLayer2METPhi;
    ConcurrentMonitorElement stage2CaloLayer2ETTRank;
    ConcurrentMonitorElement stage2CaloLayer2MHTRank;
    ConcurrentMonitorElement stage2CaloLayer2MHTPhi;
    ConcurrentMonitorElement stage2CaloLayer2HTTRank;
    ConcurrentMonitorElement stage2CaloLayer2METHFRank;
    ConcurrentMonitorElement stage2CaloLayer2METHFPhi;
    // ConcurrentMonitorElement stage2CaloLayer2ETTHFRank;
    ConcurrentMonitorElement stage2CaloLayer2MHTHFRank;
    ConcurrentMonitorElement stage2CaloLayer2MHTHFPhi;
    // ConcurrentMonitorElement stage2CaloLayer2HTTHFRank;
    ConcurrentMonitorElement stage2CaloLayer2MinBiasHFP0;
    ConcurrentMonitorElement stage2CaloLayer2MinBiasHFM0;
    ConcurrentMonitorElement stage2CaloLayer2MinBiasHFP1;
    ConcurrentMonitorElement stage2CaloLayer2MinBiasHFM1;
    ConcurrentMonitorElement stage2CaloLayer2ETTEMRank;
    ConcurrentMonitorElement stage2CaloLayer2TowCount;
  };
}

class L1TStage2CaloLayer2 : public DQMGlobalEDAnalyzer<calolayer2dqm::Histograms> {
  
 public:
  
  L1TStage2CaloLayer2(const edm::ParameterSet & ps);
  ~L1TStage2CaloLayer2() override;

 protected:

  void dqmAnalyze(const edm::Event& e, const edm::EventSetup& c, calolayer2dqm::Histograms const&) const override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&, calolayer2dqm::Histograms&) const override;
  void bookHistograms(DQMStore::ConcurrentBooker&, const edm::Run&, const edm::EventSetup&, calolayer2dqm::Histograms&) const override ;

 private:

  std::string monitorDir_;

  edm::EDGetTokenT<l1t::JetBxCollection> stage2CaloLayer2JetToken_;
  edm::EDGetTokenT<l1t::EGammaBxCollection> stage2CaloLayer2EGammaToken_;
  edm::EDGetTokenT<l1t::TauBxCollection> stage2CaloLayer2TauToken_;
  edm::EDGetTokenT<l1t::EtSumBxCollection> stage2CaloLayer2EtSumToken_;

  bool verbose_;
};

#endif 
