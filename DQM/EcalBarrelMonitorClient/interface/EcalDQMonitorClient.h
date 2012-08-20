#ifndef EcalDQMonitorClient_H
#define EcalDQMonitorClient_H

#include "DQM/EcalCommon/interface/EcalDQMonitor.h"

namespace edm{
  class ParameterSet;
  class Run;
  class LuminosityBlock;
  class Event;
  class EventSetup;
  class ConfigurationDescriptions;
}

namespace ecaldqm{
  class DQWorkerClient;
}

class EcalDQMonitorClient : public EcalDQMonitor {
 public:
  EcalDQMonitorClient(const edm::ParameterSet &);
  ~EcalDQMonitorClient();

  static void fillDescriptions(edm::ConfigurationDescriptions &);

 private:
  void beginRun(const edm::Run&, const edm::EventSetup&);
  void endRun(const edm::Run&, const edm::EventSetup&);

  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);

  void analyze(const edm::Event&, const edm::EventSetup&) {}

  void runWorkers();

  // list of workers
  std::vector<ecaldqm::DQWorker*> workers_;
  int lumiStatus_;
};

#endif
