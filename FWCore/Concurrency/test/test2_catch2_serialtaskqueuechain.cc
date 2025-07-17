//
//  SerialTaskQueue_test.cpp
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
#include <iostream>
#include "oneapi/tbb/task.h"
#include "FWCore/Concurrency/interface/SerialTaskQueueChain.h"

using namespace std::chrono_literals;

namespace {
  void join_thread(std::thread* iThread) {
    if (iThread->joinable()) {
      iThread->join();
    }
  }
}  // namespace

TEST_CASE("SerialTaskQueueChain", "[SerialTaskQueueChain]") {
  SECTION("push") {
    std::atomic<unsigned int> count{0};
    std::vector<std::shared_ptr<edm::SerialTaskQueue>> queues = {std::make_shared<edm::SerialTaskQueue>(),
                                                                 std::make_shared<edm::SerialTaskQueue>()};
    edm::SerialTaskQueueChain chain(queues);
    {
      oneapi::tbb::task_group group;
      std::atomic<int> waiting{3};
      chain.push(group, [&count, &waiting] {
        REQUIRE(count++ == 0u);
        std::this_thread::sleep_for(10us);
        --waiting;
      });
      chain.push(group, [&count, &waiting] {
        REQUIRE(count++ == 1u);
        std::this_thread::sleep_for(10us);
        --waiting;
      });
      chain.push(group, [&count, &waiting] {
        REQUIRE(count++ == 2u);
        std::this_thread::sleep_for(10us);
        --waiting;
      });
      do {
        group.wait();
      } while (0 != waiting.load());
      REQUIRE(count == 3u);
      while (chain.outstandingTasks() != 0)
        ;
    }
  }

  SECTION("push one") {
    std::atomic<unsigned int> count{0};
    std::vector<std::shared_ptr<edm::SerialTaskQueue>> queues = {std::make_shared<edm::SerialTaskQueue>()};
    edm::SerialTaskQueueChain chain(queues);
    {
      oneapi::tbb::task_group group;
      std::atomic<int> waiting{3};
      chain.push(group, [&count, &waiting] {
        REQUIRE(count++ == 0u);
        std::this_thread::sleep_for(10us);
        --waiting;
      });
      chain.push(group, [&count, &waiting] {
        REQUIRE(count++ == 1u);
        std::this_thread::sleep_for(10us);
        --waiting;
      });
      chain.push(group, [&count, &waiting] {
        REQUIRE(count++ == 2u);
        std::this_thread::sleep_for(10us);
        --waiting;
      });
      do {
        group.wait();
      } while (0 != waiting.load());
      REQUIRE(count == 3u);
      while (chain.outstandingTasks() != 0)
        ;
    }
  }

  SECTION("stress test") {
    std::vector<std::shared_ptr<edm::SerialTaskQueue>> queues = {std::make_shared<edm::SerialTaskQueue>(),
                                                                 std::make_shared<edm::SerialTaskQueue>()};
    edm::SerialTaskQueueChain chain(queues);
    unsigned int index = 100;
    const unsigned int nTasks = 1000;
    while (0 != --index) {
      oneapi::tbb::task_group group;
      std::atomic<int> waiting{2};
      std::atomic<unsigned int> count{0};
      std::atomic<bool> waitToStart{true};
      {
        std::thread pushThread([&chain, &waitToStart, &waiting, &group, &count] {
          while (waitToStart.load()) {
          };
          for (unsigned int i = 0; i < nTasks; ++i) {
            ++waiting;
            chain.push(group, [&count, &waiting] {
              ++count;
              --waiting;
            });
          }
          --waiting;
        });
        waitToStart = false;
        for (unsigned int i = 0; i < nTasks; ++i) {
          ++waiting;
          chain.push(group, [&count, &waiting] {
            ++count;
            --waiting;
          });
        }
        --waiting;
        std::shared_ptr<std::thread>(&pushThread, join_thread);
      }
      do {
        group.wait();
      } while (0 != waiting.load());
      REQUIRE(2 * nTasks == count);
    }
    while (chain.outstandingTasks() != 0)
      ;
  }
}
