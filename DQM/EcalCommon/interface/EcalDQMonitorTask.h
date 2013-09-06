#ifndef EcalDQMonitorTask_H
#define EcalDQMonitorTask_H

#include <map>

#include "DQM/EcalCommon/interface/EcalDQMonitor.h"
#include "DQM/EcalCommon/interface/Collections.h"

#include "FWCore/Utilities/interface/InputTag.h"

namespace edm{
  class ParameterSet;
  class Run;
  class LuminosityBlock;
  class Event;
  class EventSetup;
  class ConfigurationDescriptions;
}

namespace ecaldqm{
  class DQWorkerTask;
}

class EcalDQMonitorTask : public EcalDQMonitor {
 public:
  EcalDQMonitorTask(edm::ParameterSet const&);
  ~EcalDQMonitorTask();

  static void fillDescriptions(edm::ConfigurationDescriptions &);

 private:
  void beginRun(edm::Run const&, edm::EventSetup const&);
  void endRun(edm::Run const&, edm::EventSetup const&);

  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

  void analyze(edm::Event const&, edm::EventSetup const&);

  typedef void (EcalDQMonitorTask::*Processor)(edm::Event const&, ecaldqm::Collections);

  void registerCollection(ecaldqm::Collections, edm::InputTag const&);
  template <class C> void runOnCollection(edm::Event const&, ecaldqm::Collections);

  void formSchedule_(std::vector<ecaldqm::Collections> const&, std::multimap<ecaldqm::Collections, ecaldqm::Collections> const&);

  int ievt_;
  // list of workers
  std::vector<ecaldqm::DQWorkerTask*> workers_;
  // list of EDGetTokens
  edm::EDGetToken collectionTokens_[ecaldqm::nCollections];
  // schedule of collections to run
  std::vector<std::pair<Processor, ecaldqm::Collections> > schedule_;
  // which worker runs on each collection? this information is static within a job
  std::vector<ecaldqm::DQWorkerTask*> taskLists_[ecaldqm::nCollections];
  // which worker is enabled for this event?
  std::map<ecaldqm::DQWorkerTask*, bool> enabled_;

  std::map<ecaldqm::DQWorkerTask*, double> taskTimes_;
  bool evaluateTime_;

  bool allowMissingCollections_;
};

#endif
