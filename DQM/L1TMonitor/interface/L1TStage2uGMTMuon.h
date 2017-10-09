#ifndef DQM_L1TMonitor_L1TStage2uGMTMuon_h
#define DQM_L1TMonitor_L1TStage2uGMTMuon_h


#include "DataFormats/L1Trigger/interface/Muon.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class L1TStage2uGMTMuon : public DQMEDAnalyzer {

 public:

  L1TStage2uGMTMuon(const edm::ParameterSet& ps);
  virtual ~L1TStage2uGMTMuon();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 protected:

  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

 private:  

  edm::EDGetTokenT<l1t::MuonBxCollection> ugmtMuonToken;
  std::string monitorDir;
  std::string titlePrefix;
  bool verbose;

  MonitorElement* ugmtMuonBX;
  MonitorElement* ugmtnMuons;
  MonitorElement* ugmtMuonhwPt;
  MonitorElement* ugmtMuonhwEta;
  MonitorElement* ugmtMuonhwPhi;
  MonitorElement* ugmtMuonhwEtaAtVtx;
  MonitorElement* ugmtMuonhwPhiAtVtx;
  MonitorElement* ugmtMuonhwCharge;
  MonitorElement* ugmtMuonhwChargeValid;
  MonitorElement* ugmtMuonhwQual;

  MonitorElement* ugmtMuonPt;
  MonitorElement* ugmtMuonEta;
  MonitorElement* ugmtMuonPhi;
  MonitorElement* ugmtMuonEtaAtVtx;
  MonitorElement* ugmtMuonPhiAtVtx;
  MonitorElement* ugmtMuonCharge;

  MonitorElement* ugmtMuonPtvsEta;
  MonitorElement* ugmtMuonPtvsPhi;
  MonitorElement* ugmtMuonPhivsEta;
  MonitorElement* ugmtMuonPtvsEtaAtVtx;
  MonitorElement* ugmtMuonPtvsPhiAtVtx;
  MonitorElement* ugmtMuonPhiAtVtxvsEtaAtVtx;

  MonitorElement* ugmtMuonBXvshwPt;
  MonitorElement* ugmtMuonBXvshwEta;
  MonitorElement* ugmtMuonBXvshwPhi;
  MonitorElement* ugmtMuonBXvshwEtaAtVtx;
  MonitorElement* ugmtMuonBXvshwPhiAtVtx;
  MonitorElement* ugmtMuonBXvshwCharge;
  MonitorElement* ugmtMuonBXvshwChargeValid;
  MonitorElement* ugmtMuonBXvshwQual;
};

#endif
