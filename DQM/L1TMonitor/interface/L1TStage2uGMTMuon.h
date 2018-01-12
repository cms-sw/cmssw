#ifndef DQM_L1TMonitor_L1TStage2uGMTMuon_h
#define DQM_L1TMonitor_L1TStage2uGMTMuon_h


#include "DataFormats/L1Trigger/interface/Muon.h"

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace ugmtmuondqm {
  struct Histograms {
    ConcurrentMonitorElement ugmtMuonBX;
    ConcurrentMonitorElement ugmtnMuons;
    ConcurrentMonitorElement ugmtMuonhwPt;
    ConcurrentMonitorElement ugmtMuonhwEta;
    ConcurrentMonitorElement ugmtMuonhwPhi;
    ConcurrentMonitorElement ugmtMuonhwEtaAtVtx;
    ConcurrentMonitorElement ugmtMuonhwPhiAtVtx;
    ConcurrentMonitorElement ugmtMuonhwCharge;
    ConcurrentMonitorElement ugmtMuonhwChargeValid;
    ConcurrentMonitorElement ugmtMuonhwQual;

    ConcurrentMonitorElement ugmtMuonPt;
    ConcurrentMonitorElement ugmtMuonEta;
    ConcurrentMonitorElement ugmtMuonPhi;
    ConcurrentMonitorElement ugmtMuonEtaAtVtx;
    ConcurrentMonitorElement ugmtMuonPhiAtVtx;
    ConcurrentMonitorElement ugmtMuonCharge;

    ConcurrentMonitorElement ugmtMuonPtvsEta;
    ConcurrentMonitorElement ugmtMuonPtvsPhi;
    ConcurrentMonitorElement ugmtMuonPhivsEta;
    ConcurrentMonitorElement ugmtMuonPtvsEtaAtVtx;
    ConcurrentMonitorElement ugmtMuonPtvsPhiAtVtx;
    ConcurrentMonitorElement ugmtMuonPhiAtVtxvsEtaAtVtx;

    ConcurrentMonitorElement ugmtMuonBXvshwPt;
    ConcurrentMonitorElement ugmtMuonBXvshwEta;
    ConcurrentMonitorElement ugmtMuonBXvshwPhi;
    ConcurrentMonitorElement ugmtMuonBXvshwEtaAtVtx;
    ConcurrentMonitorElement ugmtMuonBXvshwPhiAtVtx;
    ConcurrentMonitorElement ugmtMuonBXvshwCharge;
    ConcurrentMonitorElement ugmtMuonBXvshwChargeValid;
    ConcurrentMonitorElement ugmtMuonBXvshwQual;
  };
}

class L1TStage2uGMTMuon : public DQMGlobalEDAnalyzer<ugmtmuondqm::Histograms> {

 public:

  L1TStage2uGMTMuon(const edm::ParameterSet& ps);
  ~L1TStage2uGMTMuon() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 protected:

  void dqmBeginRun(const edm::Run&, const edm::EventSetup&, ugmtmuondqm::Histograms &) const override;
  void bookHistograms(DQMStore::ConcurrentBooker&, const edm::Run&, const edm::EventSetup&, ugmtmuondqm::Histograms &) const override;
  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, ugmtmuondqm::Histograms const&) const override;

 private:  

  edm::EDGetTokenT<l1t::MuonBxCollection> ugmtMuonToken;
  std::string monitorDir;
  std::string titlePrefix;
  bool verbose;
  bool makeMuonAtVtxPlots;
};

#endif
