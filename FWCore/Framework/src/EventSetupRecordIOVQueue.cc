// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordIOVQueue
//
//
// Author:      W. David Dagenhart
// Created:     22 Feb 2019

#include "FWCore/Framework/interface/EventSetupRecordIOVQueue.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <exception>

namespace edm {
  namespace eventsetup {

    EventSetupRecordIOVQueue::EventSetupRecordIOVQueue(unsigned int nConcurrentIOVs)
        : iovQueue_(nConcurrentIOVs),
          isAvailable_(nConcurrentIOVs),
          // start valid cacheIdentifiers at 1 and only increment them
          // so that the EventSetup system will never return the value 0
          // as a cacheIdentifier. Then clients which store a cache identifier
          // identifying the state of their own cache can store a 0 when their
          // cache is uninitialized.
          cacheIdentifier_(1) {
      for (auto& i : isAvailable_) {
        i.store(true);
      }
    }

    EventSetupRecordIOVQueue::~EventSetupRecordIOVQueue() { assert(endIOVCalled_); }

    void EventSetupRecordIOVQueue::endIOVAsync(edm::WaitingTaskHolder iEndTask) {
      endIOVTasks_.reset();
      if (endIOVTaskHolder_.hasTask()) {
        endIOVTasks_.add(std::move(iEndTask));
        {
          WaitingTaskHolder tmp{std::move(endIOVTaskHolder_)};
        }
      }
      endIOVCalled_ = true;
    }

    bool EventSetupRecordIOVQueue::setValidityIntervalFor(const IOVSyncValue& iTime) {
      //If still valid for the same interval, then we don't need to do anything.
      if (validityInterval_.first() != IOVSyncValue::invalidIOVSyncValue() && validityInterval_.validFor(iTime)) {
        assert(firstForCurrentIOV_ == validityInterval_.first());
        intervalStatus_ = IntervalStatus::SameInterval;
        return false;
      }

      // Implementation for setting validity interval
      updateValidityIntervalAndStatus(iTime);
      if (intervalStatus_ == IntervalStatus::Invalid) {
        firstForCurrentIOV_ = IOVSyncValue::invalidIOVSyncValue();
        newIOVNeeded_ = false;
        return true;
      }
      newIOVNeeded_ = validityInterval_.first() != firstForCurrentIOV_;
      if (newIOVNeeded_) {
        firstForCurrentIOV_ = validityInterval_.first();
      }
      return newIOVNeeded_;
    }

    void EventSetupRecordIOVQueue::updateValidityIntervalAndStatus(const IOVSyncValue& iTime) {
      intervalStatus_ = IntervalStatus::Invalid;

      auto finder = recordProvider_->finder();
      if (finder.get() == nullptr) {
        return;
      }
      IOVSyncValue oldFirst(validityInterval_.first());
      IOVSyncValue oldLast(validityInterval_.last());
      validityInterval_ = finder->findIntervalFor(key(), iTime);

      // An interval is valid if and only if the start of the interval is
      // valid. If the start is valid and the end is invalid, it means we
      // do not know when the interval ends, but the interval is valid and
      // iTime is within the interval.
      if (validityInterval_.first() != IOVSyncValue::invalidIOVSyncValue()) {
        // An interval is new if the start of the interval changes
        if (validityInterval_.first() != oldFirst) {
          intervalStatus_ = IntervalStatus::NewInterval;

          // If the start is the same but the end changes, we consider
          // this the same interval because we do not want to do the
          // work to update the caches of data in this case.
        } else if (validityInterval_.last() != oldLast) {
          intervalStatus_ = IntervalStatus::UpdateIntervalEnd;
        } else {
          intervalStatus_ = IntervalStatus::SameInterval;
        }
      }
    }

    void EventSetupRecordIOVQueue::checkForNewIOVsAndStartIfNeededAsync(WaitingTaskHolder const& taskToStartAfterIOVInit,
                                                                        WaitingTaskList& endIOVWaitingTasks,
                                                                        bool newEventSetupImpl,
                                                                        EventSetupImpl& eventSetupImpl) {
      if (newIOVNeeded_) {
        newIOVNeeded_ = false;
        startNewIOVAsync(taskToStartAfterIOVInit, endIOVWaitingTasks, eventSetupImpl);
        return;
      }
      if (intervalStatus_ == IntervalStatus::Invalid) {
        return;
      }

      endIOVWaitingTasks.add(endIOVTaskHolder_);
      edm::ValidityInterval const* validityInterval = nullptr;
      if (intervalStatus_ == IntervalStatus::UpdateIntervalEnd) {
        validityInterval = &validityInterval_;
      }
      edm::EventSetupImpl* esImpl = &eventSetupImpl;
      recordProvider_->continueIOV(lastUsedIovIndex_, validityInterval, esImpl);
    }

    void EventSetupRecordIOVQueue::startNewIOVAsync(WaitingTaskHolder const& taskToStartAfterIOVInit,
                                                    WaitingTaskList& endIOVWaitingTasks,
                                                    EventSetupImpl& eventSetupImpl) {
      ++cacheIdentifier_;
      {
        // Let the old IOV end when all the lumis using it are done.
        // otherwise we'll deadlock when there is only one thread.
        WaitingTaskHolder tmp{std::move(endIOVTaskHolder_)};
      }
      auto taskHolder = std::make_shared<WaitingTaskHolder>(taskToStartAfterIOVInit);
      auto startIOVForRecord =
          [this, taskHolder, &endIOVWaitingTasks, &eventSetupImpl](edm::LimitedTaskQueue::Resumer iResumer) mutable {
            // Caught exception is propagated via WaitingTaskHolder
            CMS_SA_ALLOW try {
              unsigned int iovIndex = 0;
              auto nConcurrentIOVs = isAvailable_.size();
              for (; iovIndex < nConcurrentIOVs; ++iovIndex) {
                bool expected = true;
                if (isAvailable_[iovIndex].compare_exchange_strong(expected, false)) {
                  break;
                }
              }
              // Should never fail, just a sanity check
              if (iovIndex == nConcurrentIOVs) {
                throw edm::Exception(edm::errors::LogicError)
                    << "EventSetupRecordIOVQueue::startNewIOVAsync\n"
                    << "Couldn't find available IOV slot. This should never happen.\n"
                    << "Contact a Framework Developer\n";
              }
              lastUsedIovIndex_ = iovIndex;
              recordProvider_->initializeForNewIOV(iovIndex, cacheIdentifier_, validityInterval_, eventSetupImpl);

              auto endIOVWaitingTask = make_waiting_task([this, iResumer, iovIndex](std::exception_ptr const*) mutable {
                recordProvider_->endIOV(iovIndex);
                isAvailable_[iovIndex].store(true);
                iResumer.resume();
                endIOVTasks_.doneWaiting(std::exception_ptr());
                // There is nothing in this task to catch an exception
                // because it is extremely unlikely to throw.
              });
              endIOVTaskHolder_ = WaitingTaskHolder{*taskHolder->group(), endIOVWaitingTask};
              endIOVWaitingTasks.add(endIOVTaskHolder_);
            } catch (...) {
              taskHolder->doneWaiting(std::current_exception());
              return;
            }
            taskHolder->doneWaiting(std::exception_ptr{});
          };
      iovQueue_.pushAndPause(*taskToStartAfterIOVInit.group(), std::move(startIOVForRecord));
    }

    void EventSetupRecordIOVQueue::addRecProvider(EventSetupRecordProvider* recProvider) {
      recordProvider_ = recProvider;
    }

    void EventSetupRecordIOVQueue::reset() {
      firstForCurrentIOV_ = IOVSyncValue();
      // Force a new IOV to start with a new cacheIdentifier
      // on the next eventSetupForInstance call.
      validityInterval_ = ValidityInterval{};
      auto finder = recordProvider_->finder();
      if (finder.get() != nullptr) {
        finder->resetInterval(key());
      }
    }

  }  // namespace eventsetup
}  // namespace edm
