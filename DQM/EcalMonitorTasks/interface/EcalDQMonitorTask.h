#ifndef EcalDQMonitorTask_H
#define EcalDQMonitorTask_H

#include "DQM/EcalCommon/interface/EcalDQMonitor.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DQWorkerTask.h"
#include "Collections.h"

#include <set>
#include <map>

namespace edm
{
  class InputTag;
  class ParameterSetDescription;
}

class EcalDQMonitorTask : public DQMEDAnalyzer, public ecaldqm::EcalDQMonitor {
 public:
  EcalDQMonitorTask(edm::ParameterSet const&);
  ~EcalDQMonitorTask() {}

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;

 private:
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  
  typedef void (EcalDQMonitorTask::*Processor)(edm::Event const&, ecaldqm::Collections, std::set<ecaldqm::DQWorker*> const&);

  template <typename CollectionClass> void runOnCollection(edm::Event const&, ecaldqm::Collections, std::set<ecaldqm::DQWorker*> const&);

  void formSchedule(std::vector<ecaldqm::Collections> const&, edm::ParameterSet const&);

  /* DATA MEMBERS */

  edm::EDGetToken collectionTokens_[ecaldqm::nCollections]; // list of EDGetTokens
  std::vector<std::pair<Processor, ecaldqm::Collections> > schedule_; // schedule of collections to run
  bool allowMissingCollections_; // when true, skip missing collections silently
  int processedEvents_;

  /* TASK TIME PROFILING */
  bool evaluateTime_;
  std::map<ecaldqm::DQWorker*, double> taskTimes_;
  time_t lastResetTime_;
  float resetInterval_;
};

#endif
