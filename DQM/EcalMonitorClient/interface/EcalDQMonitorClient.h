#ifndef EcalDQMonitorClient_H
#define EcalDQMonitorClient_H

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQM/EcalCommon/interface/EcalDQMonitor.h"
#include "DQM/EcalCommon/interface/StatusManager.h"

#include "../interface/DQWorkerClient.h"

class EcalDQMonitorClient : public DQMEDHarvester, public ecaldqm::EcalDQMonitor {
 public:
  EcalDQMonitorClient(edm::ParameterSet const&);
  ~EcalDQMonitorClient();
  
  static void fillDescriptions(edm::ConfigurationDescriptions&);

 private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void dqmEndLuminosityBlock(DQMStore::IBooker&, DQMStore::IGetter&, edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
  
  void runWorkers(DQMStore::IGetter&, ecaldqm::DQWorkerClient::ProcessType);

  unsigned eventCycleLength_;
  unsigned iEvt_;

  ecaldqm::StatusManager statusManager_;
};

#endif
