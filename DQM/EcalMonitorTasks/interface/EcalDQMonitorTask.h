#ifndef EcalDQMonitorTask_H
#define EcalDQMonitorTask_H

#include "DQM/EcalCommon/interface/EcalDQMonitor.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "DQWorkerTask.h"
#include "Collections.h"

#include <set>
#include <map>

namespace edm {
  class InputTag;
  class ParameterSetDescription;
}  // namespace edm

class EcalDQMonitorTask : public DQMOneEDAnalyzer<edm::LuminosityBlockCache<ecaldqm::EcalLSCache>>,
                          public ecaldqm::EcalDQMonitor {
public:
  EcalDQMonitorTask(edm::ParameterSet const&);
  ~EcalDQMonitorTask() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;

private:
  void dqmEndRun(edm::Run const&, edm::EventSetup const&) override;
  std::shared_ptr<ecaldqm::EcalLSCache> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                                   edm::EventSetup const&) const override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;

  typedef void (EcalDQMonitorTask::*Processor)(edm::Event const&,
                                               ecaldqm::Collections,
                                               std::set<ecaldqm::DQWorker*> const&);

  template <typename CollectionClass>
  void runOnCollection(edm::Event const&, ecaldqm::Collections, std::set<ecaldqm::DQWorker*> const&);

  void formSchedule(std::vector<ecaldqm::Collections> const&, edm::ParameterSet const&);

  /* DATA MEMBERS */

  edm::EDGetToken collectionTokens_[ecaldqm::nCollections];           // list of EDGetTokens
  std::vector<std::pair<Processor, ecaldqm::Collections>> schedule_;  // schedule of collections to run

  std::vector<std::string> skipCollections_;  // list of collections to explicitly remove from schedule
                                              // note: skipping a collection here will result in
                                              // EcalDQMonitorTask not consuming it, which may lead the
                                              // module producing the collection to not run at all

  bool allowMissingCollections_;  // when true (default), skip missing collections and log as warning
                                  // note: collections skipped by the parameter skipCollections will
                                  // bypass this check and not issue warnings

  int processedEvents_;

  /* TASK TIME PROFILING */
  time_t lastResetTime_;
  float resetInterval_;
};

#endif
