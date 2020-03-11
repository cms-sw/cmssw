// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordIOVQueue
//
//
// Author:      W. David Dagenhart
// Created:     22 Feb 2019

#include "FWCore/Framework/src/EventSetupRecordIOVQueue.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"
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
      waitForIOVsInFlight_ = edm::make_empty_waiting_task();
      waitForIOVsInFlight_->increment_ref_count();
      waitForIOVsInFlight_->increment_ref_count();
    }

    EventSetupRecordIOVQueue::~EventSetupRecordIOVQueue() {
      { WaitingTaskHolder tmp{std::move(endIOVTaskHolder_)}; }
      waitForIOVsInFlight_->decrement_ref_count();
      waitForIOVsInFlight_->wait_for_all();
    }

    void EventSetupRecordIOVQueue::setNewIntervalForAnySubProcess() {
      bool newIntervalForAnySubProcess = false;
      for (auto const& recordProvider : recordProviders_) {
        if (recordProvider->intervalStatus() == EventSetupRecordProvider::IntervalStatus::NewInterval) {
          newIntervalForAnySubProcess = true;
          break;
        }
      }
      for (auto& recordProvider : recordProviders_) {
        recordProvider->setNewIntervalForAnySubProcess(newIntervalForAnySubProcess);
      }
    }

    void EventSetupRecordIOVQueue::checkForNewIOVs(WaitingTaskHolder const& taskToStartAfterIOVInit,
                                                   WaitingTaskList& endIOVWaitingTasks,
                                                   bool newEventSetupImpl) {
      for (auto& recordProvider : recordProviders_) {
        if (recordProvider->newIntervalForAnySubProcess()) {
          startNewIOVAsync(taskToStartAfterIOVInit, endIOVWaitingTasks);
          return;
        }
      }

      for (auto& recordProvider : recordProviders_) {
        if (recordProvider->intervalStatus() != EventSetupRecordProvider::IntervalStatus::Invalid) {
          endIOVWaitingTasks.add(endIOVWaitingTask_);
          break;
        }
      }

      for (auto& recordProvider : recordProviders_) {
        recordProvider->continueIOV(newEventSetupImpl);
      }
    }

    void EventSetupRecordIOVQueue::startNewIOVAsync(WaitingTaskHolder const& taskToStartAfterIOVInit,
                                                    WaitingTaskList& endIOVWaitingTasks) {
      ++cacheIdentifier_;
      {
        // Let the old IOV end when all the lumis using it are done.
        // otherwise we'll deadlock when there is only one thread.
        WaitingTaskHolder tmp{std::move(endIOVTaskHolder_)};
      }
      auto taskHolder = std::make_shared<WaitingTaskHolder>(taskToStartAfterIOVInit);
      auto startIOVForRecord =
          [this, taskHolder, &endIOVWaitingTasks](edm::LimitedTaskQueue::Resumer iResumer) mutable {
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
              for (auto recordProvider : recordProviders_) {
                recordProvider->initializeForNewIOV(iovIndex, cacheIdentifier_);
              }

              // Needed so the EventSetupRecordIOVQueue destructor knows when
              // it can run.
              waitForIOVsInFlight_->increment_ref_count();

              endIOVWaitingTask_ = make_waiting_task(tbb::task::allocate_root(),
                                                     [this, iResumer, iovIndex](std::exception_ptr const*) mutable {
                                                       for (auto recordProvider : recordProviders_) {
                                                         recordProvider->endIOV(iovIndex);
                                                       }
                                                       isAvailable_[iovIndex].store(true);
                                                       iResumer.resume();
                                                       waitForIOVsInFlight_->decrement_ref_count();
                                                       // There is nothing in this task to catch an exception
                                                       // because it is extremely unlikely to throw.
                                                     });
              endIOVTaskHolder_ = WaitingTaskHolder{endIOVWaitingTask_};
              endIOVWaitingTasks.add(endIOVWaitingTask_);
            } catch (...) {
              taskHolder->doneWaiting(std::current_exception());
              return;
            }
            taskHolder->doneWaiting(std::exception_ptr{});
          };
      iovQueue_.pushAndPause(std::move(startIOVForRecord));
    }

    void EventSetupRecordIOVQueue::addRecProvider(EventSetupRecordProvider* recProvider) {
      recordProviders_.push_back(recProvider);
    }

  }  // namespace eventsetup
}  // namespace edm
