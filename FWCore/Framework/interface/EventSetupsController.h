// -*- C++ -*-
#ifndef FWCore_Framework_EventSetupsController_h
#define FWCore_Framework_EventSetupsController_h
//
// Package:     Framework
// Class  :     EventSetupsController
//
/** \class edm::eventsetup::EventSetupsController

 Description: Manage work related to concurrent IOVs in
 the EventSetup system.

 Usage: EventProcessor owns one of these through a unique_ptr.

*/
//
// Original Authors:  Chris Jones, David Dagenhart
//          Created:  Wed Jan 12 14:30:42 CST 2011
//

#include <memory>
#include <vector>

#include "oneapi/tbb/task_group.h"

#include "FWCore/Framework/interface/NumberOfConcurrentIOVs.h"
#include "FWCore/Utilities/interface/propagate_const.h"

namespace edm {

  class ActivityRegistry;
  class EventSetupImpl;
  class ParameterSet;
  class IOVSyncValue;
  class ModuleTypeResolverMaker;
  class SerialTaskQueue;
  class ServiceToken;
  class WaitingTaskHolder;
  class WaitingTaskList;

  namespace eventsetup {

    class EventSetupProvider;
    class EventSetupRecordIOVQueue;

    class EventSetupsController {
    public:
      EventSetupsController();
      explicit EventSetupsController(ModuleTypeResolverMaker const* resolverMaker);

      EventSetupsController(EventSetupsController const&) = delete;
      EventSetupsController const& operator=(EventSetupsController const&) = delete;
      ~EventSetupsController();

      void endIOVsAsync(edm::WaitingTaskHolder iEndTask);

      std::shared_ptr<EventSetupProvider> makeProvider(ParameterSet&,
                                                       ActivityRegistry*,
                                                       ParameterSet const* eventSetupPset = nullptr,
                                                       unsigned int maxConcurrentIOVs = 0,
                                                       bool dumpOptions = false);

      void finishConfiguration();

      // The main purpose of this function is to call eventSetupForInstanceAsync. It might
      // be called immediately or we might need to wait until all the currently active
      // IOVs end. If there is an exception, then a signal is emitted and the exception
      // is propagated.
      void runOrQueueEventSetupForInstanceAsync(IOVSyncValue const&,
                                                WaitingTaskHolder& taskToStartAfterIOVInit,
                                                WaitingTaskList& endIOVWaitingTasks,
                                                std::shared_ptr<const EventSetupImpl>&,
                                                edm::SerialTaskQueue& queueWhichWaitsForIOVsToFinish,
                                                ActivityRegistry*,
                                                ServiceToken const&,
                                                bool iForceCacheClear = false);

      // Pass in an IOVSyncValue to let the EventSetup system know which run and lumi
      // need to be processed and prepare IOVs for it (also could be a time or only a run).
      // Pass in a WaitingTaskHolder that allows the EventSetup to communicate when all
      // the IOVs are ready to process this IOVSyncValue. Note this preparation is often
      // done in asynchronous tasks and the function might return before all the preparation
      // is complete.
      // Pass in endIOVWaitingTasks, additions to this WaitingTaskList allow the lumi or
      // run to notify the EventSetup system when a lumi or run transition is done and no
      // longer needs its EventSetup IOVs.
      // Pass in a reference to a shared_ptr<EventSetupImpl> that gets set and is used
      // to give clients access to the EventSetup system such that for each record the IOV
      // associated with the correct IOVSyncValue will be used.
      void eventSetupForInstanceAsync(IOVSyncValue const&,
                                      WaitingTaskHolder const& taskToStartAfterIOVInit,
                                      WaitingTaskList& endIOVWaitingTasks,
                                      std::shared_ptr<const EventSetupImpl>&);

      bool doWeNeedToWaitForIOVsToFinish(IOVSyncValue const&) const;

      void forceCacheClear();

      bool hasNonconcurrentFinder() const { return hasNonconcurrentFinder_; }
      bool mustFinishConfiguration() const { return mustFinishConfiguration_; }

    private:
      void initializeEventSetupRecordIOVQueues();

      // ---------- member data --------------------------------
      propagate_const<std::shared_ptr<EventSetupProvider>> provider_;
      NumberOfConcurrentIOVs numberOfConcurrentIOVs_;

      // This data member is intentionally declared after provider_
      // It is important that this is destroyed first.
      std::vector<propagate_const<std::unique_ptr<EventSetupRecordIOVQueue>>> eventSetupRecordIOVQueues_;

      ModuleTypeResolverMaker const* typeResolverMaker_ = nullptr;

      bool hasNonconcurrentFinder_ = false;
      bool mustFinishConfiguration_ = true;
    };

    void synchronousEventSetupForInstance(IOVSyncValue const& syncValue,
                                          oneapi::tbb::task_group& iGroup,
                                          eventsetup::EventSetupsController& espController);
  }  // namespace eventsetup
}  // namespace edm
#endif
