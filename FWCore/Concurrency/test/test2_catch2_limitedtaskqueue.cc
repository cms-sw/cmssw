//
//  LimitedTaskQueue_test.cpp
//  DispatchProcessingDemo
//
//  Created by Chris Jones on 9/27/11.
//

#include <iostream>
#include <catch.hpp>
#include <chrono>
#include <memory>
#include <atomic>
#include <thread>
#include "oneapi/tbb/task_arena.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/LimitedTaskQueue.h"
#include "FWCore/Concurrency/interface/FunctorTask.h"

using namespace std::chrono_literals;

TEST_CASE("LimitedTaskQueue", "[LimitedTaskQueue]") {
  SECTION("push") {
    {
      std::atomic<unsigned int> count{0};
      edm::LimitedTaskQueue queue{1};
      {
        std::atomic<int> waitingTasks{3};
        oneapi::tbb::task_group group;
        queue.push(group, [&count, &waitingTasks] {
          REQUIRE(count++ == 0u);
          std::this_thread::sleep_for(10us);
          --waitingTasks;
        });
        queue.push(group, [&count, &waitingTasks] {
          REQUIRE(count++ == 1u);
          std::this_thread::sleep_for(10us);
          --waitingTasks;
        });
        queue.push(group, [&count, &waitingTasks] {
          REQUIRE(count++ == 2u);
          std::this_thread::sleep_for(10us);
          --waitingTasks;
        });
        do {
          group.wait();
        } while (0 != waitingTasks.load());
        REQUIRE(count == 3u);
      }
    }
    {
      std::atomic<unsigned int> count{0};
      constexpr unsigned int kMax = 2;
      edm::LimitedTaskQueue queue{kMax};
      {
        std::atomic<int> waitingTasks{3};
        oneapi::tbb::task_group group;
        queue.push(group, [&count, &waitingTasks, kMax] {
          REQUIRE(count++ < kMax);
          std::this_thread::sleep_for(10us);
          --count;
          --waitingTasks;
        });
        queue.push(group, [&count, &waitingTasks, kMax] {
          REQUIRE(count++ < kMax);
          std::this_thread::sleep_for(10us);
          --count;
          --waitingTasks;
        });
        queue.push(group, [&count, &waitingTasks, kMax] {
          REQUIRE(count++ < kMax);
          std::this_thread::sleep_for(10us);
          --count;
          --waitingTasks;
        });
        do {
          group.wait();
        } while (0 != waitingTasks);
        REQUIRE(count == 0u);
      }
    }
  }

  SECTION("pause") {
    std::atomic<unsigned int> count{0};
    edm::LimitedTaskQueue queue{1};
    {
      {
        std::atomic<int> waitingTasks{3};
        oneapi::tbb::task_group group;
        edm::LimitedTaskQueue::Resumer resumer;
        std::atomic<bool> resumerSet{false};
        std::exception_ptr e1;
        queue.pushAndPause(
            group, [&resumer, &resumerSet, &count, &waitingTasks, &e1](edm::LimitedTaskQueue::Resumer iResumer) {
              resumer = std::move(iResumer);
              resumerSet = true;
              try {
                REQUIRE(++count == 1u);
              } catch (...) {
                e1 = std::current_exception();
              }
              --waitingTasks;
            });
        std::exception_ptr e2;
        queue.push(group, [&count, &waitingTasks, &e2] {
          try {
            REQUIRE(++count == 2u);
          } catch (...) {
            e2 = std::current_exception();
          }
          --waitingTasks;
        });
        std::exception_ptr e3;
        queue.push(group, [&count, &waitingTasks, &e3] {
          try {
            REQUIRE(++count == 3u);
          } catch (...) {
            e3 = std::current_exception();
          }
          --waitingTasks;
        });
        std::this_thread::sleep_for(100us);
        REQUIRE(2u >= count);
        while (not resumerSet) {
        }
        REQUIRE(resumer.resume());
        do {
          group.wait();
        } while (0 != waitingTasks.load());
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
    //catch2 REQUIRE is not thread safe
    std::mutex mutex;
    while (0 != --index) {
      std::atomic<int> waiting{1};
      std::atomic<unsigned int> count{0};
      std::atomic<unsigned int> nRunningTasks{0};
      std::atomic<bool> waitToStart{true};
      {
        group.run([&queue, &waitToStart, &group, &waiting, &count, &nRunningTasks, &mutex, kMax] {
          while (waitToStart) {
          }
          for (unsigned int i = 0; i < nTasks; ++i) {
            ++waiting;
            queue.push(group, [&count, &waiting, &nRunningTasks, &mutex, kMax] {
              std::shared_ptr<std::atomic<int>> guardAgain{&waiting, [](auto* v) { --(*v); }};
              auto nrt = nRunningTasks++;
              if (nrt >= kMax) {
                std::cout << "ERROR " << nRunningTasks << " >= " << kMax << std::endl;
                std::lock_guard lock{mutex};
                REQUIRE(nrt < kMax);
              }
              ++count;
              --nRunningTasks;
            });
          }
        });
        group.run([&queue, &waitToStart, &group, &waiting, &count, &nRunningTasks, &mutex, kMax] {
          waitToStart = false;
          for (unsigned int i = 0; i < nTasks; ++i) {
            ++waiting;
            queue.push(group, [&count, &waiting, &nRunningTasks, &mutex, kMax] {
              std::shared_ptr<std::atomic<int>> guardAgain{&waiting, [](auto* v) { --(*v); }};
              auto nrt = nRunningTasks++;
              if (nrt >= kMax) {
                std::cout << "ERROR " << nRunningTasks << " >= " << kMax << std::endl;
                std::lock_guard lock{mutex};
                REQUIRE(nrt < kMax);
              }
              ++count;
              --nRunningTasks;
            });
          }
          --waiting;
        });
      }
      do {
        group.wait();
      } while (0 != waiting.load());
      REQUIRE(nRunningTasks == 0u);
      REQUIRE(2 * nTasks == count);
    }
  }
}
