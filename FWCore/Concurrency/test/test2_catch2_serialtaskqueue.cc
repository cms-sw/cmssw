//
//  SerialTaskQueue_test.cpp
//  DispatchProcessingDemo
//
//  Created by Chris Jones on 9/27/11.
//

#include <iostream>
#define CATCH_CONFIG_MAIN

#include <catch.hpp>
#include <chrono>
#include <memory>
#include <atomic>
#include <thread>
#include "oneapi/tbb/task_arena.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"
#include "FWCore/Concurrency/interface/FunctorTask.h"

using namespace std::chrono_literals;

TEST_CASE("SerialTaskQueue", "[SerialTaskQueue]") {
  SECTION("push") {
    std::atomic<unsigned int> count{0};
    edm::SerialTaskQueue queue;
    {
      std::atomic<unsigned int> waitingTasks{3};
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

  SECTION("pause") {
    std::atomic<unsigned int> count{0};
    edm::SerialTaskQueue queue;
    {
      queue.pause();
      {
        std::atomic<unsigned int> waitingTasks{1};
        oneapi::tbb::task_group group;
        queue.push(group, [&count, &waitingTasks] {
          REQUIRE(count++ == 0u);
          --waitingTasks;
        });
        std::this_thread::sleep_for(1000us);
        REQUIRE(count == 0u);
        queue.resume();
        do {
          group.wait();
        } while (0 != waitingTasks.load());
        REQUIRE(count == 1u);
      }
      {
        std::atomic<unsigned int> waitingTasks{3};
        oneapi::tbb::task_group group;
        queue.push(group, [&count, &queue, &waitingTasks] {
          queue.pause();
          REQUIRE(count++ == 1u);
          --waitingTasks;
        });
        queue.push(group, [&count, &waitingTasks] {
          REQUIRE(count++ == 2u);
          --waitingTasks;
        });
        queue.push(group, [&count, &waitingTasks] {
          REQUIRE(count++ == 3u);
          --waitingTasks;
        });
        std::this_thread::sleep_for(100us);
        REQUIRE(2u >= count);
        queue.resume();
        do {
          group.wait();
        } while (0 != waitingTasks.load());
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
      std::atomic<unsigned int> waitingTasks{2};
      std::atomic<unsigned int> count{0};
      std::atomic<bool> waitToStart{true};
      {
        group.run([&queue, &waitToStart, &waitingTasks, &count, &group] {
          while (waitToStart.load()) {
          }
          for (unsigned int i = 0; i < nTasks; ++i) {
            ++waitingTasks;
            queue.push(group, [&count, &waitingTasks] {
              ++count;
              --waitingTasks;
            });
          }
          --waitingTasks;
        });
        group.run([&queue, &waitToStart, &waitingTasks, &count, &group] {
          waitToStart = false;
          for (unsigned int i = 0; i < nTasks; ++i) {
            ++waitingTasks;
            queue.push(group, [&count, &waitingTasks] {
              ++count;
              --waitingTasks;
            });
          }
          --waitingTasks;
        });
      }
      do {
        group.wait();
      } while (0 != waitingTasks.load());
      REQUIRE(2 * nTasks == count);
    }
  }
}
