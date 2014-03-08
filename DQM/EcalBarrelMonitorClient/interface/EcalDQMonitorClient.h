#ifndef EcalDQMonitorClient_H
#define EcalDQMonitorClient_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQM/EcalCommon/interface/EcalDQMonitor.h"
#include "DQM/EcalCommon/interface/StatusManager.h"

#include "../interface/DQWorkerClient.h"

class EcalDQMonitorClient : public edm::EDAnalyzer, public ecaldqm::EcalDQMonitor {
 public:
  EcalDQMonitorClient(edm::ParameterSet const&);
  ~EcalDQMonitorClient() {}

  static void fillDescriptions(edm::ConfigurationDescriptions&);

 private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  
  void runWorkers(ecaldqm::DQWorkerClient::ProcessType);

  unsigned eventCycleLength_;
  unsigned iEvt_;

  ecaldqm::StatusManager statusManager_;
};

#endif
