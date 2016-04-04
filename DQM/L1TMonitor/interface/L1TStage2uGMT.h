#ifndef DQM_L1TMonitor_L1TStage2uGMT_h
#define DQM_L1TMonitor_L1TStage2uGMT_h


#include "DataFormats/L1Trigger/interface/Muon.h"

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

  edm::EDGetTokenT<l1t::MuonBxCollection> ugmtToken;
  std::string monitorDir;
  bool verbose;

  MonitorElement* ugmtBX;
  MonitorElement* ugmtPt;
  MonitorElement* ugmtEta;
  MonitorElement* ugmtPhi;
  MonitorElement* ugmtCharge;
  MonitorElement* ugmtChargeValid;
  MonitorElement* ugmtQual;
  MonitorElement* ugmtIso;

  MonitorElement* ugmtBXvsPt;
  MonitorElement* ugmtBXvsEta;
  MonitorElement* ugmtBXvsPhi;
  MonitorElement* ugmtBXvsCharge;
  MonitorElement* ugmtBXvsChargeValid;
  MonitorElement* ugmtBXvsQual;
  MonitorElement* ugmtBXvsIso;

  MonitorElement* ugmtPtvsEta;
  MonitorElement* ugmtPtvsPhi;
  MonitorElement* ugmtPhivsEta;
};

#endif
