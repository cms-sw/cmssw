// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupsController
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones, W. David Dagenhart
//         Created:  Wed Jan 12 14:30:44 CST 2011
//

#include "FWCore/Framework/interface/EventSetupsController.h"

#include <exception>
#include <set>

#include "FWCore/Concurrency/interface/SerialTaskQueue.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"
#include "FWCore/Framework/src/EventSetupProviderMaker.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/EventSetupRecordIOVQueue.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/src/SendSourceTerminationSignalIfException.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace edm {
  namespace eventsetup {

    EventSetupsController::EventSetupsController() {}
    EventSetupsController::EventSetupsController(ModuleTypeResolverMaker const* resolverMaker)
        : typeResolverMaker_(resolverMaker) {}

    EventSetupsController::~EventSetupsController() {}

    void EventSetupsController::endIOVsAsync(edm::WaitingTaskHolder iEndTask) {
      for (auto& eventSetupRecordIOVQueue : eventSetupRecordIOVQueues_) {
        eventSetupRecordIOVQueue->endIOVAsync(iEndTask);
      }
    }

    std::shared_ptr<EventSetupProvider> EventSetupsController::makeProvider(ParameterSet& iPSet,
                                                                            ActivityRegistry* activityRegistry,
                                                                            ParameterSet const* eventSetupPset,
                                                                            unsigned int maxConcurrentIOVs,
                                                                            bool dumpOptions) {
      // Makes an EventSetupProvider
      // Also parses the prefer information from ParameterSets and puts
      // it in a map that is stored in the EventSetupProvider
      std::shared_ptr<EventSetupProvider> returnValue(makeEventSetupProvider(iPSet, activityRegistry));

      // Construct the ESProducers and ESSources.
      // shared_ptrs to them are stored in the EventSetupProvider
      fillEventSetupProvider(typeResolverMaker_, *returnValue, iPSet);

      numberOfConcurrentIOVs_.readConfigurationParameters(eventSetupPset, maxConcurrentIOVs, dumpOptions);

      provider_ = returnValue;
      return returnValue;
    }

    void EventSetupsController::finishConfiguration() {
      if (mustFinishConfiguration_) {
        numberOfConcurrentIOVs_.fillRecordsNotAllowingConcurrentIOVs(*provider_);
        provider_->finishConfiguration(numberOfConcurrentIOVs_, hasNonconcurrentFinder_);
        provider_->clearInitializationData();
        provider_->updateLookup();

        initializeEventSetupRecordIOVQueues();
        numberOfConcurrentIOVs_.clear();
        mustFinishConfiguration_ = false;
      }
    }

    void EventSetupsController::runOrQueueEventSetupForInstanceAsync(
        IOVSyncValue const& iSync,
        WaitingTaskHolder& taskToStartAfterIOVInit,
        WaitingTaskList& endIOVWaitingTasks,
        std::shared_ptr<const EventSetupImpl>& eventSetupImpl,
        edm::SerialTaskQueue& queueWhichWaitsForIOVsToFinish,
        ActivityRegistry* actReg,
        ServiceToken const& iToken,
        bool iForceCacheClear) {
      auto asyncEventSetup =
          [this, &endIOVWaitingTasks, &eventSetupImpl, &queueWhichWaitsForIOVsToFinish, actReg, iForceCacheClear](
              IOVSyncValue const& iSync, WaitingTaskHolder& task) {
            queueWhichWaitsForIOVsToFinish.pause();
            CMS_SA_ALLOW try {
              if (iForceCacheClear) {
                forceCacheClear();
              }
              SendSourceTerminationSignalIfException sentry(actReg);
              {
                //all EventSetupRecordIntervalFinders are sequentially set to the
                // new SyncValue in the call. The async part is just waiting for
                // the Records to be available which is done after the SyncValue setup.
                actReg->preESSyncIOVSignal_.emit(iSync);
                auto postSignal = [&iSync](ActivityRegistry* actReg) { actReg->postESSyncIOVSignal_.emit(iSync); };
                std::unique_ptr<ActivityRegistry, decltype(postSignal)> guard(actReg, postSignal);
                eventSetupForInstanceAsync(iSync, task, endIOVWaitingTasks, eventSetupImpl);
              }
              sentry.completedSuccessfully();
            } catch (...) {
              task.doneWaiting(std::current_exception());
            }
          };
      if (doWeNeedToWaitForIOVsToFinish(iSync) || iForceCacheClear) {
        // We get inside this block if there is an EventSetup
        // module not able to handle concurrent IOVs (usually an ESSource)
        // and the new sync value is outside the current IOV of that module.
        // Also at beginRun when forcing caches to clear.
        auto group = taskToStartAfterIOVInit.group();
        ServiceWeakToken weakToken = iToken;
        queueWhichWaitsForIOVsToFinish.push(*group,
                                            [iSync, taskToStartAfterIOVInit, asyncEventSetup, weakToken]() mutable {
                                              ServiceRegistry::Operate operate(weakToken.lock());
                                              asyncEventSetup(iSync, taskToStartAfterIOVInit);
                                            });
      } else {
        asyncEventSetup(iSync, taskToStartAfterIOVInit);
      }
    }

    void EventSetupsController::eventSetupForInstanceAsync(IOVSyncValue const& syncValue,
                                                           WaitingTaskHolder const& taskToStartAfterIOVInit,
                                                           WaitingTaskList& endIOVWaitingTasks,
                                                           std::shared_ptr<const EventSetupImpl>& eventSetupImpl) {
      finishConfiguration();

      bool newEventSetupImpl = false;
      eventSetupImpl.reset();

      provider_->setAllValidityIntervals(syncValue);

      for (auto& eventSetupRecordIOVQueue : eventSetupRecordIOVQueues_) {
        eventSetupRecordIOVQueue->setNewInterval();
      }

      // Decides whether we can reuse the existing EventSetupImpl and if we can
      // returns it. If a new one is needed it will create it, although the pointers
      // to the EventSetupRecordImpl's will not be set yet in the returned EventSetupImpl
      // object.
      eventSetupImpl = provider_->eventSetupForInstance(syncValue, newEventSetupImpl);

      for (auto& eventSetupRecordIOVQueue : eventSetupRecordIOVQueues_) {
        eventSetupRecordIOVQueue->checkForNewIOVs(taskToStartAfterIOVInit, endIOVWaitingTasks, newEventSetupImpl);
      }
    }

    bool EventSetupsController::doWeNeedToWaitForIOVsToFinish(IOVSyncValue const& syncValue) const {
      if (hasNonconcurrentFinder()) {
        if (provider_->doWeNeedToWaitForIOVsToFinish(syncValue)) {
          return true;
        }
      }
      return false;
    }

    void EventSetupsController::forceCacheClear() { provider_->forceCacheClear(); }

    void EventSetupsController::initializeEventSetupRecordIOVQueues() {
      std::set<EventSetupRecordKey> keys;
      provider_->fillKeys(keys);

      for (auto const& key : keys) {
        eventSetupRecordIOVQueues_.push_back(
            std::make_unique<EventSetupRecordIOVQueue>(numberOfConcurrentIOVs_.numberOfConcurrentIOVs(key)));
        EventSetupRecordIOVQueue& iovQueue = *eventSetupRecordIOVQueues_.back();
        EventSetupRecordProvider* recProvider = provider_->tryToGetRecordProvider(key);
        if (recProvider) {
          iovQueue.addRecProvider(recProvider);
        }
      }
    }

    void synchronousEventSetupForInstance(IOVSyncValue const& syncValue,
                                          oneapi::tbb::task_group& iGroup,
                                          eventsetup::EventSetupsController& espController) {
      FinalWaitingTask waitUntilIOVInitializationCompletes{iGroup};

      // These do nothing ...
      WaitingTaskList dummyWaitingTaskList;
      std::shared_ptr<const EventSetupImpl> dummyEventSetupImpl;

      {
        WaitingTaskHolder waitingTaskHolder(iGroup, &waitUntilIOVInitializationCompletes);
        // Caught exception is propagated via WaitingTaskHolder
        CMS_SA_ALLOW try {
          // All the real work is done here.
          espController.eventSetupForInstanceAsync(
              syncValue, waitingTaskHolder, dummyWaitingTaskList, dummyEventSetupImpl);
          dummyWaitingTaskList.doneWaiting(std::exception_ptr{});
        } catch (...) {
          dummyWaitingTaskList.doneWaiting(std::exception_ptr{});
          waitingTaskHolder.doneWaiting(std::current_exception());
        }
      }
      waitUntilIOVInitializationCompletes.wait();
    }
  }  // namespace eventsetup
}  // namespace edm
