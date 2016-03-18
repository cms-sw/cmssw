#ifndef DQM_L1TMonitor_L1TStage2uGMT_h
#define DQM_L1TMonitor_L1TStage2uGMT_h


#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class L1TStage2uGMT : public DQMEDAnalyzer {

 public:

  L1TStage2uGMT(const edm::ParameterSet& ps);
  virtual ~L1TStage2uGMT();

 protected:

  virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&);
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  virtual void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

 private:  

  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> ugmtBMTFToken;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> ugmtOMTFToken;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> ugmtEMTFToken;
  edm::EDGetTokenT<l1t::MuonBxCollection> ugmtMuonToken;
  std::string monitorDir;
  bool verbose;

  MonitorElement* ugmtBMTFhwPt;
  MonitorElement* ugmtBMTFhwEta;
  MonitorElement* ugmtBMTFhwPhi;
  MonitorElement* ugmtBMTFhwSign;
  MonitorElement* ugmtBMTFhwSignValid;
  MonitorElement* ugmtBMTFhwQual;
  MonitorElement* ugmtBMTFlink;

  MonitorElement* ugmtOMTFhwPt;
  MonitorElement* ugmtOMTFhwEta;
  MonitorElement* ugmtOMTFhwPhi;
  MonitorElement* ugmtOMTFhwSign;
  MonitorElement* ugmtOMTFhwSignValid;
  MonitorElement* ugmtOMTFhwQual;
  MonitorElement* ugmtOMTFlink;

  MonitorElement* ugmtEMTFhwPt;
  MonitorElement* ugmtEMTFhwEta;
  MonitorElement* ugmtEMTFhwPhi;
  MonitorElement* ugmtEMTFhwSign;
  MonitorElement* ugmtEMTFhwSignValid;
  MonitorElement* ugmtEMTFhwQual;
  MonitorElement* ugmtEMTFlink;

  MonitorElement* ugmtBMTFBX;
  MonitorElement* ugmtOMTFBX;
  MonitorElement* ugmtEMTFBX;
  MonitorElement* ugmtLinkBX;

  MonitorElement* ugmtMuonBX;
  MonitorElement* ugmtMuonhwPt;
  MonitorElement* ugmtMuonhwEta;
  MonitorElement* ugmtMuonhwPhi;
  MonitorElement* ugmtMuonhwCharge;
  MonitorElement* ugmtMuonhwChargeValid;
  MonitorElement* ugmtMuonhwQual;
  MonitorElement* ugmtMuonhwIso;

  MonitorElement* ugmtMuonhwPtvshwEta;
  MonitorElement* ugmtMuonhwPtvshwPhi;
  MonitorElement* ugmtMuonhwPhivshwEta;

  MonitorElement* ugmtMuonPt;
  MonitorElement* ugmtMuonEta;
  MonitorElement* ugmtMuonPhi;
  MonitorElement* ugmtMuonCharge;

  MonitorElement* ugmtMuonBXvshwPt;
  MonitorElement* ugmtMuonBXvshwEta;
  MonitorElement* ugmtMuonBXvshwPhi;
  MonitorElement* ugmtMuonBXvshwCharge;
  MonitorElement* ugmtMuonBXvshwChargeValid;
  MonitorElement* ugmtMuonBXvshwQual;
  MonitorElement* ugmtMuonBXvshwIso;
};

#endif
