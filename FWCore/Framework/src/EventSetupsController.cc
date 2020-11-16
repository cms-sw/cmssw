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

#include "FWCore/Framework/src/EventSetupsController.h"

#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/src/EventSetupProviderMaker.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/ParameterSetIDHolder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <algorithm>
#include <iostream>
#include <set>

namespace edm {
  namespace eventsetup {

    EventSetupsController::EventSetupsController() {}

    void EventSetupsController::endIOVs() {
      for (auto& eventSetupRecordIOVQueue : eventSetupRecordIOVQueues_) {
        eventSetupRecordIOVQueue->endIOV();
      }
    }

    std::shared_ptr<EventSetupProvider> EventSetupsController::makeProvider(ParameterSet& iPSet,
                                                                            ActivityRegistry* activityRegistry,
                                                                            ParameterSet const* eventSetupPset) {
      // Makes an EventSetupProvider
      // Also parses the prefer information from ParameterSets and puts
      // it in a map that is stored in the EventSetupProvider
      std::shared_ptr<EventSetupProvider> returnValue(
          makeEventSetupProvider(iPSet, providers_.size(), activityRegistry));

      // Construct the ESProducers and ESSources
      // shared_ptrs to them are temporarily stored in this
      // EventSetupsController and in the EventSetupProvider
      fillEventSetupProvider(*this, *returnValue, iPSet);

      numberOfConcurrentIOVs_.readConfigurationParameters(eventSetupPset);

      providers_.push_back(returnValue);
      return returnValue;
    }

    void EventSetupsController::setMaxConcurrentIOVs(unsigned int nStreams, unsigned int nConcurrentLumis) {
      numberOfConcurrentIOVs_.setMaxConcurrentIOVs(nStreams, nConcurrentLumis);
    }

    void EventSetupsController::finishConfiguration() {
      if (mustFinishConfiguration_) {
        for (auto& eventSetupProvider : providers_) {
          numberOfConcurrentIOVs_.fillRecordsNotAllowingConcurrentIOVs(*eventSetupProvider);
        }

        for (auto& eventSetupProvider : providers_) {
          eventSetupProvider->finishConfiguration(numberOfConcurrentIOVs_, hasNonconcurrentFinder_);
        }

        // When the ESSources and ESProducers were constructed a first pass was
        // done which attempts to get component sharing between SubProcesses
        // correct, but in this pass only the configuration of the components
        // being shared are compared. This is not good enough for ESProducers.
        // In the following function, all the other components that contribute
        // to the same record and also the records that record depends on are
        // also checked. The component sharing is appropriately fixed as necessary.
        checkESProducerSharing();
        clearComponents();

        initializeEventSetupRecordIOVQueues();
        numberOfConcurrentIOVs_.clear();
        mustFinishConfiguration_ = false;
      }
    }

    void EventSetupsController::eventSetupForInstance(
        IOVSyncValue const& syncValue,
        WaitingTaskHolder const& taskToStartAfterIOVInit,
        WaitingTaskList& endIOVWaitingTasks,
        std::vector<std::shared_ptr<const EventSetupImpl>>& eventSetupImpls) {
      finishConfiguration();

      bool newEventSetupImpl = false;
      eventSetupImpls.clear();
      eventSetupImpls.reserve(providers_.size());

      // Note that unless there are one or more SubProcesses providers_ will only
      // contain one element.

      for (auto& eventSetupProvider : providers_) {
        eventSetupProvider->setAllValidityIntervals(syncValue);
      }

      for (auto& eventSetupRecordIOVQueue : eventSetupRecordIOVQueues_) {
        // For a particular record, if the top level process or any SubProcess requires
        // starting a new IOV, then we must start a new IOV for all of them. And we
        // need to know whether this is needed at this point in time. This is
        // recorded in the EventSetupRecordProviders.
        eventSetupRecordIOVQueue->setNewIntervalForAnySubProcess();
      }

      for (auto& eventSetupProvider : providers_) {
        // Decides whether we can reuse the existing EventSetupImpl and if we can
        // returns it. If a new one is needed it will create it, although the pointers
        // to the EventSetupRecordImpl's will not be set yet in the returned EventSetupImpl
        // object.
        eventSetupImpls.push_back(eventSetupProvider->eventSetupForInstance(syncValue, newEventSetupImpl));
      }

      for (auto& eventSetupRecordIOVQueue : eventSetupRecordIOVQueues_) {
        eventSetupRecordIOVQueue->checkForNewIOVs(taskToStartAfterIOVInit, endIOVWaitingTasks, newEventSetupImpl);
      }
    }

    void EventSetupsController::eventSetupForInstance(IOVSyncValue const& syncValue) {
      // This function only supports use cases where the event setup
      // system is used without multiple concurrent IOVs.
      // At the time this comment is being written, this is used for
      // run transitions and in unit test code. In the future,
      // it may only be needed for unit tests. This function uses
      // the other version of eventSetupForInstance that
      // supports concurrent IOVs. To get this to work, a couple
      // arguments to that function need dummy objects that do
      // not serve any purpose in this context. We also need to
      // add in a task to wait for the asynchronous initialization
      // of IOVs to complete.

      auto waitUntilIOVInitializationCompletes = make_empty_waiting_task();
      waitUntilIOVInitializationCompletes->increment_ref_count();

      // These do nothing ...
      WaitingTaskList dummyWaitingTaskList;
      std::vector<std::shared_ptr<const EventSetupImpl>> dummyEventSetupImpls;

      {
        WaitingTaskHolder waitingTaskHolder(waitUntilIOVInitializationCompletes.get());
        // Caught exception is propagated via WaitingTaskHolder
        CMS_SA_ALLOW try {
          // All the real work is done here.
          eventSetupForInstance(syncValue, waitingTaskHolder, dummyWaitingTaskList, dummyEventSetupImpls);
          dummyWaitingTaskList.doneWaiting(std::exception_ptr{});
        } catch (...) {
          dummyWaitingTaskList.doneWaiting(std::exception_ptr{});
          waitingTaskHolder.doneWaiting(std::current_exception());
        }
      }
      waitUntilIOVInitializationCompletes->wait_for_all();

      if (waitUntilIOVInitializationCompletes->exceptionPtr() != nullptr) {
        std::rethrow_exception(*(waitUntilIOVInitializationCompletes->exceptionPtr()));
      }
    }

    bool EventSetupsController::doWeNeedToWaitForIOVsToFinish(IOVSyncValue const& syncValue) const {
      if (hasNonconcurrentFinder()) {
        for (auto& eventSetupProvider : providers_) {
          if (eventSetupProvider->doWeNeedToWaitForIOVsToFinish(syncValue)) {
            return true;
          }
        }
      }
      return false;
    }

    void EventSetupsController::forceCacheClear() {
      for (auto& eventSetupProvider : providers_) {
        eventSetupProvider->forceCacheClear();
      }
    }

    std::shared_ptr<DataProxyProvider> EventSetupsController::getESProducerAndRegisterProcess(
        ParameterSet const& pset, unsigned subProcessIndex) {
      // Try to find a DataProxyProvider with a matching ParameterSet
      auto elements = esproducers_.equal_range(pset.id());
      for (auto it = elements.first; it != elements.second; ++it) {
        // Untracked parameters must also match, do complete comparison if IDs match
        if (isTransientEqual(pset, *it->second.pset())) {
          // Register processes with an exact match
          it->second.subProcessIndexes().push_back(subProcessIndex);
          // Return the DataProxyProvider
          return it->second.provider();
        }
      }
      // Could not find it
      return std::shared_ptr<DataProxyProvider>();
    }

    void EventSetupsController::putESProducer(ParameterSet const& pset,
                                              std::shared_ptr<DataProxyProvider> const& component,
                                              unsigned subProcessIndex) {
      auto newElement =
          esproducers_.insert(std::pair<ParameterSetID, ESProducerInfo>(pset.id(), ESProducerInfo(&pset, component)));
      // Register processes with an exact match
      newElement->second.subProcessIndexes().push_back(subProcessIndex);
    }

    std::shared_ptr<EventSetupRecordIntervalFinder> EventSetupsController::getESSourceAndRegisterProcess(
        ParameterSet const& pset, unsigned subProcessIndex) {
      // Try to find a EventSetupRecordIntervalFinder with a matching ParameterSet
      auto elements = essources_.equal_range(pset.id());
      for (auto it = elements.first; it != elements.second; ++it) {
        // Untracked parameters must also match, do complete comparison if IDs match
        if (isTransientEqual(pset, *it->second.pset())) {
          // Register processes with an exact match
          it->second.subProcessIndexes().push_back(subProcessIndex);
          // Return the EventSetupRecordIntervalFinder
          return it->second.finder();
        }
      }
      // Could not find it
      return std::shared_ptr<EventSetupRecordIntervalFinder>();
    }

    void EventSetupsController::putESSource(ParameterSet const& pset,
                                            std::shared_ptr<EventSetupRecordIntervalFinder> const& component,
                                            unsigned subProcessIndex) {
      auto newElement =
          essources_.insert(std::pair<ParameterSetID, ESSourceInfo>(pset.id(), ESSourceInfo(&pset, component)));
      // Register processes with an exact match
      newElement->second.subProcessIndexes().push_back(subProcessIndex);
    }

    void EventSetupsController::clearComponents() {
      esproducers_.clear();
      essources_.clear();
    }

    void EventSetupsController::lookForMatches(ParameterSetID const& psetID,
                                               unsigned subProcessIndex,
                                               unsigned precedingProcessIndex,
                                               bool& firstProcessWithThisPSet,
                                               bool& precedingHasMatchingPSet) const {
      auto elements = esproducers_.equal_range(psetID);
      for (auto it = elements.first; it != elements.second; ++it) {
        std::vector<unsigned> const& subProcessIndexes = it->second.subProcessIndexes();

        auto iFound = std::find(subProcessIndexes.begin(), subProcessIndexes.end(), subProcessIndex);
        if (iFound == subProcessIndexes.end()) {
          continue;
        }

        if (iFound == subProcessIndexes.begin()) {
          firstProcessWithThisPSet = true;
          precedingHasMatchingPSet = false;
        } else {
          auto iFoundPreceding = std::find(subProcessIndexes.begin(), iFound, precedingProcessIndex);
          if (iFoundPreceding == iFound) {
            firstProcessWithThisPSet = false;
            precedingHasMatchingPSet = false;
          } else {
            firstProcessWithThisPSet = false;
            precedingHasMatchingPSet = true;
          }
        }
        return;
      }
      throw edm::Exception(edm::errors::LogicError) << "EventSetupsController::lookForMatches\n"
                                                    << "Subprocess index not found. This should never happen\n"
                                                    << "Please report this to a Framework Developer\n";
    }

    bool EventSetupsController::isFirstMatch(ParameterSetID const& psetID,
                                             unsigned subProcessIndex,
                                             unsigned precedingProcessIndex) const {
      auto elements = esproducers_.equal_range(psetID);
      for (auto it = elements.first; it != elements.second; ++it) {
        std::vector<unsigned> const& subProcessIndexes = it->second.subProcessIndexes();

        auto iFound = std::find(subProcessIndexes.begin(), subProcessIndexes.end(), subProcessIndex);
        if (iFound == subProcessIndexes.end()) {
          continue;
        }

        auto iFoundPreceding = std::find(subProcessIndexes.begin(), iFound, precedingProcessIndex);
        if (iFoundPreceding == iFound) {
          break;
        } else {
          return iFoundPreceding == subProcessIndexes.begin();
        }
      }
      throw edm::Exception(edm::errors::LogicError) << "EventSetupsController::isFirstMatch\n"
                                                    << "Subprocess index not found. This should never happen\n"
                                                    << "Please report this to a Framework Developer\n";
      return false;
    }

    bool EventSetupsController::isLastMatch(ParameterSetID const& psetID,
                                            unsigned subProcessIndex,
                                            unsigned precedingProcessIndex) const {
      auto elements = esproducers_.equal_range(psetID);
      for (auto it = elements.first; it != elements.second; ++it) {
        std::vector<unsigned> const& subProcessIndexes = it->second.subProcessIndexes();

        auto iFound = std::find(subProcessIndexes.begin(), subProcessIndexes.end(), subProcessIndex);
        if (iFound == subProcessIndexes.end()) {
          continue;
        }

        auto iFoundPreceding = std::find(subProcessIndexes.begin(), iFound, precedingProcessIndex);
        if (iFoundPreceding == iFound) {
          break;
        } else {
          return (++iFoundPreceding) == iFound;
        }
      }
      throw edm::Exception(edm::errors::LogicError) << "EventSetupsController::isLastMatch\n"
                                                    << "Subprocess index not found. This should never happen\n"
                                                    << "Please report this to a Framework Developer\n";
      return false;
    }

    bool EventSetupsController::isMatchingESSource(ParameterSetID const& psetID,
                                                   unsigned subProcessIndex,
                                                   unsigned precedingProcessIndex) const {
      auto elements = essources_.equal_range(psetID);
      for (auto it = elements.first; it != elements.second; ++it) {
        std::vector<unsigned> const& subProcessIndexes = it->second.subProcessIndexes();

        auto iFound = std::find(subProcessIndexes.begin(), subProcessIndexes.end(), subProcessIndex);
        if (iFound == subProcessIndexes.end()) {
          continue;
        }

        auto iFoundPreceding = std::find(subProcessIndexes.begin(), iFound, precedingProcessIndex);
        if (iFoundPreceding == iFound) {
          return false;
        } else {
          return true;
        }
      }
      throw edm::Exception(edm::errors::LogicError) << "EventSetupsController::lookForMatchingESSource\n"
                                                    << "Subprocess index not found. This should never happen\n"
                                                    << "Please report this to a Framework Developer\n";
      return false;
    }

    bool EventSetupsController::isMatchingESProducer(ParameterSetID const& psetID,
                                                     unsigned subProcessIndex,
                                                     unsigned precedingProcessIndex) const {
      auto elements = esproducers_.equal_range(psetID);
      for (auto it = elements.first; it != elements.second; ++it) {
        std::vector<unsigned> const& subProcessIndexes = it->second.subProcessIndexes();

        auto iFound = std::find(subProcessIndexes.begin(), subProcessIndexes.end(), subProcessIndex);
        if (iFound == subProcessIndexes.end()) {
          continue;
        }

        auto iFoundPreceding = std::find(subProcessIndexes.begin(), iFound, precedingProcessIndex);
        if (iFoundPreceding == iFound) {
          return false;
        } else {
          return true;
        }
      }
      throw edm::Exception(edm::errors::LogicError) << "EventSetupsController::lookForMatchingESSource\n"
                                                    << "Subprocess index not found. This should never happen\n"
                                                    << "Please report this to a Framework Developer\n";
      return false;
    }

    ParameterSet const* EventSetupsController::getESProducerPSet(ParameterSetID const& psetID,
                                                                 unsigned subProcessIndex) const {
      auto elements = esproducers_.equal_range(psetID);
      for (auto it = elements.first; it != elements.second; ++it) {
        std::vector<unsigned> const& subProcessIndexes = it->second.subProcessIndexes();

        auto iFound = std::find(subProcessIndexes.begin(), subProcessIndexes.end(), subProcessIndex);
        if (iFound == subProcessIndexes.end()) {
          continue;
        }
        return it->second.pset();
      }
      throw edm::Exception(edm::errors::LogicError) << "EventSetupsController::getESProducerPSet\n"
                                                    << "Subprocess index not found. This should never happen\n"
                                                    << "Please report this to a Framework Developer\n";
      return nullptr;
    }

    void EventSetupsController::checkESProducerSharing() {
      // Loop over SubProcesses, skip the top level process.
      auto esProvider = providers_.begin();
      auto const esProviderEnd = providers_.end();
      if (esProvider != esProviderEnd)
        ++esProvider;
      for (; esProvider != esProviderEnd; ++esProvider) {
        // An element is added to this set for each ESProducer
        // when we have determined which preceding process
        // this process can share that ESProducer with or
        // we have determined that it cannot be shared with
        // any preceding process.
        // Note the earliest possible preceding process
        // will be the one selected if there is more than one.
        std::set<ParameterSetIDHolder> sharingCheckDone;

        // This will hold an entry for DataProxy's that are
        // referenced by an EventSetupRecord in this SubProcess.
        // But only for DataProxy's that are associated with
        // an ESProducer (not the ones associated with ESSource's
        // or EDLooper's)
        std::map<EventSetupRecordKey, std::vector<ComponentDescription const*>> referencedESProducers;

        // For each EventSetupProvider from a SubProcess, loop over the
        // EventSetupProviders from the preceding processes (the first
        // preceding process will be the top level process and the others
        // SubProcess's)
        for (auto precedingESProvider = providers_.begin(); precedingESProvider != esProvider; ++precedingESProvider) {
          (*esProvider)->checkESProducerSharing(**precedingESProvider, sharingCheckDone, referencedESProducers, *this);
        }

        (*esProvider)->resetRecordToProxyPointers();
      }
      for (auto& eventSetupProvider : providers_) {
        eventSetupProvider->clearInitializationData();
      }
    }

    void EventSetupsController::initializeEventSetupRecordIOVQueues() {
      std::set<EventSetupRecordKey> keys;
      for (auto const& provider : providers_) {
        provider->fillKeys(keys);
      }

      for (auto const& key : keys) {
        eventSetupRecordIOVQueues_.push_back(
            std::make_unique<EventSetupRecordIOVQueue>(numberOfConcurrentIOVs_.numberOfConcurrentIOVs(key)));
        EventSetupRecordIOVQueue& iovQueue = *eventSetupRecordIOVQueues_.back();
        for (auto& provider : providers_) {
          EventSetupRecordProvider* recProvider = provider->tryToGetRecordProvider(key);
          if (recProvider) {
            iovQueue.addRecProvider(recProvider);
          }
        }
      }
    }

  }  // namespace eventsetup
}  // namespace edm
