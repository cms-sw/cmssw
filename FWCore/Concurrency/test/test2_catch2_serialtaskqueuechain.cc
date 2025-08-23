//
//  SerialTaskQueue_test.cpp
//  DispatchProcessingDemo
//
//  Created by Chris Jones on 9/27/11.
//

#include <iostream>
#include <sstream>
#include <catch.hpp>
#include <chrono>
#include <memory>
#include <atomic>
#include <iostream>
#include "oneapi/tbb/task.h"
#include "oneapi/tbb/global_control.h"
#include "oneapi/tbb/task_arena.h"
#include "FWCore/Concurrency/interface/SerialTaskQueueChain.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

using namespace std::chrono_literals;

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
    REQUIRE(2 <= oneapi::tbb::this_task_arena::max_concurrency());
    oneapi::tbb::task_arena arena(2);
    arena.execute([&]() {
      edm::SerialTaskQueueChain chain(queues);
      unsigned int index = 100;
      const unsigned int nTasks = 1000;
      while (0 != --index) {
        oneapi::tbb::task_group group;
        edm::FinalWaitingTask lastTask(group);
        std::atomic<unsigned int> count{0};
        std::atomic<unsigned int> waitToStart{2};
        {
          edm::WaitingTaskHolder lastHolder(group, &lastTask);

          group.run([&chain, &waitToStart, &group, &count, lastHolder, index] {
            --waitToStart;
            while (waitToStart.load() != 0)
              ;
            std::ostringstream ss;
            ss << "start task 1, index: " << index << "\n";
            std::cout << ss.str() << std::flush;
            for (unsigned int i = 0; i < nTasks; ++i) {
              chain.push(group, [&count, lastHolder] { ++count; });
            }
            ss.str(std::string());
            ss << "stop task 1, index: " << index << "\n";
            std::cout << ss.str() << std::flush;
          });
          group.run([&chain, &waitToStart, &group, &count, lastHolder, index] {
            --waitToStart;
            while (waitToStart.load() != 0)
              ;
            std::ostringstream ss;
            ss << "start task 2, index: " << index << "\n";
            std::cout << ss.str() << std::flush;
            for (unsigned int i = 0; i < nTasks; ++i) {
              chain.push(group, [&count, lastHolder] { ++count; });
            }
            ss.str(std::string());
            ss << "stop task 2, index: " << index << "\n";
            std::cout << ss.str() << std::flush;
          });
        }
        std::cout << "Waiting for tasks to finish, index: " << index << "\n" << std::flush;
        lastTask.wait();
        REQUIRE(2 * nTasks == count);
      }
      CHECK(0 == chain.outstandingTasks());
      while (chain.outstandingTasks() != 0)
        ;
    });
  }
}