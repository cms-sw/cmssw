//
//  LimitedTaskQueue_test.cpp
//  DispatchProcessingDemo
//
//  Created by Chris Jones on 9/27/11.
//

#include <iostream>
#include <catch2/catch_all.hpp>
#include <chrono>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include "oneapi/tbb/task_arena.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/LimitedTaskQueue.h"
#include "FWCore/Concurrency/interface/FunctorTask.h"

using namespace std::chrono_literals;

namespace {
  std::mutex g_requiresMutex;

}
//catch2 REQUIRE is not thread safe
#define SAFE_REQUIRE(__var__)           \
  {                                     \
    std::lock_guard g{g_requiresMutex}; \
    REQUIRE(__var__);                   \
  }

TEST_CASE("LimitedTaskQueue", "[LimitedTaskQueue]") {
  SECTION("push") {
    {
      std::atomic<unsigned int> count{0};
      edm::LimitedTaskQueue queue{1};
      {
        oneapi::tbb::task_group group;
        edm::FinalWaitingTask lastTask(group);
        edm::WaitingTaskHolder waitingTask(group, &lastTask);
        queue.push(group, [&count, waitingTask] {
          REQUIRE(count++ == 0u);
          std::this_thread::sleep_for(10us);
        });
        queue.push(group, [&count, waitingTask] {
          REQUIRE(count++ == 1u);
          std::this_thread::sleep_for(10us);
        });
        queue.push(group, [&count, lastTask = std::move(waitingTask)] {
          REQUIRE(count++ == 2u);
          std::this_thread::sleep_for(10us);
        });
        lastTask.wait();
        REQUIRE(count == 3u);
      }
    }
    {
      std::atomic<unsigned int> count{0};
      constexpr unsigned int kMax = 2;
      edm::LimitedTaskQueue queue{kMax};
      {
        oneapi::tbb::task_group group;
        edm::FinalWaitingTask lastTask(group);
        edm::WaitingTaskHolder waitingTask(group, &lastTask);
        queue.push(group, [&count, waitingTask] {
          SAFE_REQUIRE(count++ < kMax);
          std::this_thread::sleep_for(10us);
          --count;
        });
        queue.push(group, [&count, waitingTask] {
          SAFE_REQUIRE(count++ < kMax);
          std::this_thread::sleep_for(10us);
          --count;
        });
        queue.push(group, [&count, lastTask = std::move(waitingTask)] {
          SAFE_REQUIRE(count++ < kMax);
          std::this_thread::sleep_for(10us);
          --count;
        });
        lastTask.wait();
        REQUIRE(count == 0u);
      }
    }
  }

  SECTION("pause") {
    std::atomic<unsigned int> count{0};
    edm::LimitedTaskQueue queue{1};
    {
      {
        oneapi::tbb::task_group group;
        edm::FinalWaitingTask lastTask(group);
        edm::WaitingTaskHolder waitingTask(group, &lastTask);

        edm::LimitedTaskQueue::Resumer resumer;
        std::atomic<bool> resumerSet{false};
        std::exception_ptr e1;
        queue.pushAndPause(group,
                           [&resumer, &resumerSet, &count, waitingTask, &e1](edm::LimitedTaskQueue::Resumer iResumer) {
                             resumer = std::move(iResumer);
                             resumerSet = true;
                             try {
                               SAFE_REQUIRE(++count == 1u);
                             } catch (...) {
                               e1 = std::current_exception();
                             }
                           });
        std::exception_ptr e2;
        queue.push(group, [&count, waitingTask, &e2] {
          try {
            SAFE_REQUIRE(++count == 2u);
          } catch (...) {
            e2 = std::current_exception();
          }
        });
        std::exception_ptr e3;
        queue.push(group, [&count, lastTask = std::move(waitingTask), &e3] {
          try {
            SAFE_REQUIRE(++count == 3u);
          } catch (...) {
            e3 = std::current_exception();
          }
        });
        std::this_thread::sleep_for(100us);
        REQUIRE(2u >= count);
        while (not resumerSet) {
        }
        SAFE_REQUIRE(resumer.resume());
        lastTask.wait();
        REQUIRE(count == 3u);
        if (e1) {
          std::rethrow_exception(e1);
        }
        if (e2) {
          std::rethrow_exception(e2);
        }
        if (e3) {
          std::rethrow_exception(e3);
        }
      }
    }
  }

  SECTION("stress test") {
    oneapi::tbb::task_group group;
    constexpr unsigned int kMax = 3;
    edm::LimitedTaskQueue queue{kMax};
    unsigned int index = 100;
    const unsigned int nTasks = 1000;
    while (0 != --index) {
      edm::FinalWaitingTask lastTask(group);

      std::atomic<unsigned int> count{0};
      std::atomic<unsigned int> nRunningTasks{0};
      std::atomic<bool> waitToStart{true};
      {
        edm::WaitingTaskHolder waitingTask(group, &lastTask);

        group.run([&queue, &waitToStart, &group, waitingTask, &count, &nRunningTasks] {
          while (waitToStart) {
          }
          for (unsigned int i = 0; i < nTasks; ++i) {
            queue.push(group, [&count, waitingTask, &nRunningTasks] {
              auto nrt = nRunningTasks++;
              if (nrt >= kMax) {
                std::cout << "ERROR " << nRunningTasks << " >= " << kMax << std::endl;
                SAFE_REQUIRE(nrt < kMax);
              }
              ++count;
              --nRunningTasks;
            });
          }
        });
        group.run([&queue, &waitToStart, &group, waitingTask, &count, &nRunningTasks] {
          waitToStart = false;
          for (unsigned int i = 0; i < nTasks; ++i) {
            queue.push(group, [&count, waitingTask, &nRunningTasks] {
              auto nrt = nRunningTasks++;
              if (nrt >= kMax) {
                std::cout << "ERROR " << nRunningTasks << " >= " << kMax << std::endl;
                SAFE_REQUIRE(nrt < kMax);
              }
              ++count;
              --nRunningTasks;
            });
          }
        });
      }
      lastTask.wait();
      REQUIRE(nRunningTasks == 0u);
      REQUIRE(2 * nTasks == count);
    }
  }
}
