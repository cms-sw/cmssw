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
#include <oneapi/tbb/task_arena.h>

#include "FWCore/Concurrency/interface/SerialTaskQueue.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"
#include "FWCore/Framework/src/EventSetupProviderMaker.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/EventSetupRecordIOVCoordinator.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/src/SendSourceTerminationSignalIfException.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "makeFindersForRecords.h"

namespace edm {
  namespace eventsetup {

    EventSetupsController::EventSetupsController() {}
    EventSetupsController::EventSetupsController(ModuleTypeResolverMaker const* resolverMaker)
        : typeResolverMaker_(resolverMaker) {}

    EventSetupsController::~EventSetupsController() {}

    void EventSetupsController::endIOVsAsync(edm::WaitingTaskHolder iEndTask) {
      for (auto& eventSetupRecordIOVCoordinator : eventSetupRecordIOVCoordinators_) {
        eventSetupRecordIOVCoordinator->endIOVAsync(iEndTask);
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
      loadedFinders_ = fillEventSetupProvider(typeResolverMaker_, *returnValue, iPSet);

      numberOfConcurrentIOVs_.readConfigurationParameters(eventSetupPset, maxConcurrentIOVs, dumpOptions);

      provider_ = returnValue;
      return returnValue;
    }

    void EventSetupsController::addExtra(std::shared_ptr<EventSetupRecordIntervalFinder> iFinder) {
      assert(loadedFinders_.has_value());
      loadedFinders_->push_back(iFinder);
    }
    void EventSetupsController::addExtra(std::shared_ptr<eventsetup::ESProductResolverProvider> iProvider) {
      assert(provider_);
      provider_->add(iProvider);
    }

    void EventSetupsController::finishConfiguration() {
      if (mustFinishConfiguration_) {
        numberOfConcurrentIOVs_.fillRecordsNotAllowingConcurrentIOVs(*provider_);
        std::set<edm::eventsetup::EventSetupRecordKey> finderRecords;
        for (auto const& finder : loadedFinders_.value()) {
          auto const& recordsUsing = finder->findingForRecords();
          finderRecords.insert(recordsUsing.begin(), recordsUsing.end());
        }
        provider_->finishConfiguration(finderRecords, numberOfConcurrentIOVs_);
        provider_->clearInitializationData();
        provider_->updateLookup();

        assert(loadedFinders_.has_value());
        auto findersForRecords = impl::makeFindersForRecords(provider_->keys(), loadedFinders_.value());
        initializeEventSetupRecordIOVCoordinators(findersForRecords);
        numberOfConcurrentIOVs_.clear();
        loadedFinders_.reset();
        mustFinishConfiguration_ = false;
      }
    }

    void EventSetupsController::runOrQueueEventSetupForInstanceAsync(
        IOVSyncValue const& iSync,
        WaitingTaskHolder& taskToStartAfterIOVInit,
        WaitingTaskList& endIOVWaitingTasks,
        std::shared_ptr<const EventSetupImpl>& eventSetupImpl,
        ActivityRegistry* actReg,
        ServiceToken const& iToken) {
      auto asyncEventSetup = [this, &endIOVWaitingTasks, &eventSetupImpl, actReg](IOVSyncValue const& iSync,
                                                                                  WaitingTaskHolder& task) {
        CMS_SA_ALLOW try {
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
      asyncEventSetup(iSync, taskToStartAfterIOVInit);
    }

    void EventSetupsController::eventSetupForInstanceAsync(IOVSyncValue const& syncValue,
                                                           WaitingTaskHolder const& taskToStartAfterIOVInit,
                                                           WaitingTaskList& endIOVWaitingTasks,
                                                           std::shared_ptr<const EventSetupImpl>& eventSetupImpl) {
      assert(mustFinishConfiguration_ == false);
      bool newEventSetupImpl = false;
      eventSetupImpl.reset();

      for (auto& eventSetupRecordIOVCoordinator : eventSetupRecordIOVCoordinators_) {
        if (eventSetupRecordIOVCoordinator->setValidityIntervalFor(syncValue)) {
          newEventSetupImpl = true;
        }
      }

      // Decides whether we can reuse the existing EventSetupImpl and if we can
      // returns it. If a new one is needed it will create it, although the pointers
      // to the EventSetupRecordImpl's will not be set yet in the returned EventSetupImpl
      // object.
      auto nonConst = provider_->cachedEventSetup(newEventSetupImpl);

      eventSetupImpl = nonConst;
      for (auto& eventSetupRecordIOVCoordinator : eventSetupRecordIOVCoordinators_) {
        eventSetupRecordIOVCoordinator->checkForNewIOVsAndStartIfNeededAsync(
            taskToStartAfterIOVInit, endIOVWaitingTasks, newEventSetupImpl, *nonConst);
      }
    }

    void EventSetupsController::initializeEventSetupRecordIOVCoordinators(
        std::map<edm::eventsetup::EventSetupRecordKey, std::shared_ptr<edm::EventSetupRecordIntervalFinder>> const&
            iKeyToFinders) {
      std::set<EventSetupRecordKey> keys = provider_->keys();

      for (auto const& key : keys) {
        eventSetupRecordIOVCoordinators_.push_back(
            std::make_unique<EventSetupRecordIOVCoordinator>(numberOfConcurrentIOVs_.numberOfConcurrentIOVs(key)));
        EventSetupRecordIOVCoordinator& iovCoordinator = *eventSetupRecordIOVCoordinators_.back();
        EventSetupRecordProvider* recProvider = provider_->tryToGetRecordProvider(key);
        if (recProvider) {
          iovCoordinator.addRecProvider(recProvider);
        }
        auto finderIt = iKeyToFinders.find(key);
        if (finderIt != iKeyToFinders.end()) {
          iovCoordinator.setFinder(finderIt->second);
        }
      }
    }

    void EventSetupsController::resetRecordPlusDependentRecords(EventSetupRecordKey const& recordKey) {
      auto dependentKeys = provider_->resetRecordPlusDependentRecords(recordKey);
      for (auto& queue : eventSetupRecordIOVCoordinators_) {
        if (queue->key() == recordKey) {
          queue->reset();
        } else if (std::find(dependentKeys.begin(), dependentKeys.end(), queue->key()) != dependentKeys.end()) {
          queue->reset();
        }
      }
    }

    void synchronousEventSetupForInstance(IOVSyncValue const& syncValue,
                                          oneapi::tbb::task_group& iGroup,
                                          eventsetup::EventSetupsController& espController) {
      espController.finishConfiguration();
      FinalWaitingTask waitUntilIOVInitializationCompletes{iGroup};

      // These do nothing ...

      oneapi::tbb::task_arena arena(1);
      arena.execute([&]() {
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
      });
    }
  }  // namespace eventsetup
}  // namespace edm
