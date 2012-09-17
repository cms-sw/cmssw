#ifndef EcalDQMonitorTask_H
#define EcalDQMonitorTask_H

#include <map>

#include "DQM/EcalCommon/interface/EcalDQMonitor.h"

#include "Collections.h"

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
  EcalDQMonitorTask(const edm::ParameterSet &);
  ~EcalDQMonitorTask();

  static void fillDescriptions(edm::ConfigurationDescriptions &);

 private:
  void beginRun(const edm::Run&, const edm::EventSetup&);
  void endRun(const edm::Run&, const edm::EventSetup&);

  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);

  void analyze(const edm::Event&, const edm::EventSetup&);

  typedef void (EcalDQMonitorTask::*Processor)(const edm::Event&, ecaldqm::Collections);

  template <class C> void runOnCollection(const edm::Event&, ecaldqm::Collections);

  void formSchedule_(std::vector<ecaldqm::Collections> const&);

  int processedEvents_;
  // list of InputTags
  edm::InputTag collectionTags_[ecaldqm::nCollections];
  // schedule of collections to run
  std::vector<std::pair<Processor, ecaldqm::Collections> > schedule_;
  // which worker runs on each collection? this information is static within a job
  std::vector<ecaldqm::DQWorkerTask*> taskLists_[ecaldqm::nCollections];
  // which worker is enabled for this event?
  std::map<ecaldqm::DQWorkerTask*, bool> enabled_;

  std::map<ecaldqm::DQWorkerTask*, double> taskTimes_;
  bool evaluateTime_;

  bool allowMissingCollections_;

  time_t lastResetTime_;
  float resetInterval_;
};

#endif
