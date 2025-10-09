//
//  SerialTaskQueue_test.cpp
//  DispatchProcessingDemo
//
//  Created by Chris Jones on 9/27/11.
//

#include <iostream>
#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>
#include <chrono>
#include <memory>
#include <atomic>
#include <thread>
#include "oneapi/tbb/task_arena.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"
#include "FWCore/Concurrency/interface/FunctorTask.h"

using namespace std::chrono_literals;

TEST_CASE("SerialTaskQueue", "[SerialTaskQueue]") {
  SECTION("push") {
    std::atomic<unsigned int> count{0};
    edm::SerialTaskQueue queue;
    {
      oneapi::tbb::task_group group;
      edm::FinalWaitingTask lastTask(group);
      {
        edm::WaitingTaskHolder waitingTask(group, &lastTask);
        queue.push(group, [&count, waitingTask] {
          REQUIRE(count++ == 0u);
          std::this_thread::sleep_for(10us);
        });
        queue.push(group, [&count, waitingTask] {
          REQUIRE(count++ == 1u);
          std::this_thread::sleep_for(10us);
        });
        queue.push(group, [&count, waitingTask] {
          REQUIRE(count++ == 2u);
          std::this_thread::sleep_for(10us);
        });
      }
      lastTask.wait();
      REQUIRE(count == 3u);
    }
  }

  SECTION("pause") {
    std::atomic<unsigned int> count{0};
    edm::SerialTaskQueue queue;
    {
      queue.pause();
      {
        oneapi::tbb::task_group group;
        edm::FinalWaitingTask lastTask(group);

        queue.push(group, [&count, waitingTask = edm::WaitingTaskHolder(group, &lastTask)] { REQUIRE(count++ == 0u); });
        std::this_thread::sleep_for(1000us);
        REQUIRE(count == 0u);
        queue.resume();
        lastTask.wait();
        REQUIRE(count == 1u);
      }
      {
        oneapi::tbb::task_group group;
        edm::FinalWaitingTask lastTask(group);
        {
          edm::WaitingTaskHolder waitingTask(group, &lastTask);
          queue.push(group, [&count, &queue, waitingTask] {
            queue.pause();
            REQUIRE(count++ == 1u);
          });
          queue.push(group, [&count, waitingTask] { REQUIRE(count++ == 2u); });
          queue.push(group, [&count, waitingTask] { REQUIRE(count++ == 3u); });
        }
        std::this_thread::sleep_for(100us);
        REQUIRE(2u >= count);
        queue.resume();
        lastTask.wait();
        REQUIRE(count == 4u);
      }
    }
  }

  SECTION("stress test") {
    oneapi::tbb::task_group group;
    edm::SerialTaskQueue queue;
    unsigned int index = 100;
    const unsigned int nTasks = 1000;
    while (0 != --index) {
      edm::FinalWaitingTask lastTask(group);
      std::atomic<unsigned int> count{0};
      std::atomic<bool> waitToStart{true};
      {
        edm::WaitingTaskHolder waitingTask(group, &lastTask);
        group.run([&queue, &waitToStart, waitingTask, &count, &group] {
          while (waitToStart.load()) {
          }
          for (unsigned int i = 0; i < nTasks; ++i) {
            queue.push(group, [&count, waitingTask] { ++count; });
          }
        });
        group.run([&queue, &waitToStart, waitingTask, &count, &group] {
          waitToStart = false;
          for (unsigned int i = 0; i < nTasks; ++i) {
            queue.push(group, [&count, waitingTask] { ++count; });
          }
        });
      }
      lastTask.wait();
      REQUIRE(2 * nTasks == count);
    }
  }
}
