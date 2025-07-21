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
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

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
      edm::FinalWaitingTask lastTask(group);
      {
        edm::WaitingTaskHolder lastHolder(group, &lastTask);

        chain.push(group, [&count, lastHolder] {
          REQUIRE(count++ == 0u);
          std::this_thread::sleep_for(10us);
        });
        chain.push(group, [&count, lastHolder] {
          REQUIRE(count++ == 1u);
          std::this_thread::sleep_for(10us);
        });
        chain.push(group, [&count, lastHolder] {
          REQUIRE(count++ == 2u);
          std::this_thread::sleep_for(10us);
        });
      }
      lastTask.wait();
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
      edm::FinalWaitingTask lastTask(group);

      {
        edm::WaitingTaskHolder lastHolder(group, &lastTask);
        chain.push(group, [&count, lastHolder] {
          REQUIRE(count++ == 0u);
          std::this_thread::sleep_for(10us);
        });
        chain.push(group, [&count, lastHolder] {
          REQUIRE(count++ == 1u);
          std::this_thread::sleep_for(10us);
        });
        chain.push(group, [&count, lastHolder] {
          REQUIRE(count++ == 2u);
          std::this_thread::sleep_for(10us);
        });
      }
      lastTask.wait();
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
      edm::FinalWaitingTask lastTask(group);
      std::atomic<unsigned int> count{0};
      std::atomic<bool> waitToStart{true};
      {
        edm::WaitingTaskHolder lastHolder(group, &lastTask);
        std::thread pushThread([&chain, &waitToStart, &group, &count, lastHolder] {
          while (waitToStart.load()) {
          };
          for (unsigned int i = 0; i < nTasks; ++i) {
            chain.push(group, [&count, lastHolder] { ++count; });
          }
        });
        waitToStart = false;
        for (unsigned int i = 0; i < nTasks; ++i) {
          chain.push(group, [&count, lastHolder] { ++count; });
        }
        lastHolder.doneWaiting(std::exception_ptr());
        std::shared_ptr<std::thread>(&pushThread, join_thread);
      }
      lastTask.wait();
      REQUIRE(2 * nTasks == count);
    }
    while (chain.outstandingTasks() != 0)
      ;
  }
}
