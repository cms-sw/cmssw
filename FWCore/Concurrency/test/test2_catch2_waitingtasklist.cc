//
//  WaitingTaskList_test.cpp
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
#include "oneapi/tbb/task.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"

namespace {
  class TestCalledTask : public edm::WaitingTask {
  public:
    TestCalledTask(std::atomic<bool>& iCalled, std::exception_ptr& iPtr) : m_called(iCalled), m_ptr(iPtr) {}
    void execute() final {
      if (exceptionPtr()) {
        m_ptr = exceptionPtr();
      }
      m_called = true;
      return;
    }

  private:
    std::atomic<bool>& m_called;
    std::exception_ptr& m_ptr;
  };

  class TestValueSetTask : public edm::WaitingTask {
  public:
    TestValueSetTask(std::atomic<bool>& iValue) : m_value(iValue) {}
    void execute() final {
      REQUIRE(m_value);
      return;
    }

  private:
    std::atomic<bool>& m_value;
  };

  void join_thread(std::thread* iThread) {
    if (iThread->joinable()) {
      iThread->join();
    }
  }
}  // namespace
using namespace std::chrono_literals;

TEST_CASE("WaitingTaskList", "[WaitingTaskList]") {
  SECTION("add then done") {
    std::atomic<bool> called{false};
    edm::WaitingTaskList waitList;
    {
      std::exception_ptr excPtr;
      auto t = new TestCalledTask{called, excPtr};
      oneapi::tbb::task_group group;
      waitList.add(edm::WaitingTaskHolder(group, t));
      std::this_thread::sleep_for(10us);
      REQUIRE_FALSE(called);
      waitList.doneWaiting(std::exception_ptr{});
      group.wait();
      REQUIRE(called);
      REQUIRE_FALSE(bool(excPtr));
    }
    waitList.reset();
    called = false;
    {
      std::exception_ptr excPtr;
      auto t = new TestCalledTask{called, excPtr};
      oneapi::tbb::task_group group;
      waitList.add(edm::WaitingTaskHolder(group, t));
      std::this_thread::sleep_for(10us);
      REQUIRE_FALSE(called);
      waitList.doneWaiting(std::exception_ptr{});
      group.wait();
      REQUIRE(called);
      REQUIRE_FALSE(bool(excPtr));
    }
  }

  SECTION("done then add") {
    std::atomic<bool> called{false};
    std::exception_ptr excPtr;
    edm::WaitingTaskList waitList;
    {
      oneapi::tbb::task_group group;
      auto t = new TestCalledTask{called, excPtr};
      waitList.doneWaiting(std::exception_ptr{});
      waitList.add(edm::WaitingTaskHolder(group, t));
      group.wait();
      REQUIRE(called);
      REQUIRE_FALSE(bool(excPtr));
    }
  }

  SECTION("add then done failed") {
    std::atomic<bool> called{false};
    edm::WaitingTaskList waitList;
    {
      std::exception_ptr excPtr;
      auto t = new TestCalledTask{called, excPtr};
      oneapi::tbb::task_group group;
      waitList.add(edm::WaitingTaskHolder(group, t));
      std::this_thread::sleep_for(10us);
      REQUIRE_FALSE(called);
      waitList.doneWaiting(std::make_exception_ptr(std::string("failed")));
      group.wait();
      REQUIRE(called);
      REQUIRE(bool(excPtr));
    }
  }

  SECTION("done then add failed") {
    std::atomic<bool> called{false};
    std::exception_ptr excPtr;
    edm::WaitingTaskList waitList;
    {
      auto t = new TestCalledTask{called, excPtr};
      waitList.doneWaiting(std::make_exception_ptr(std::string("failed")));
      oneapi::tbb::task_group group;
      waitList.add(edm::WaitingTaskHolder(group, t));
      group.wait();
      REQUIRE(called);
      REQUIRE(bool(excPtr));
    }
  }

  SECTION("stress Test") {
    edm::WaitingTaskList waitList;
    oneapi::tbb::task_group group;
    unsigned int index = 1000;
    const unsigned int nTasks = 10000;
    while (0 != --index) {
      edm::FinalWaitingTask waitTask{group};
      auto* pWaitTask = &waitTask;
      {
        edm::WaitingTaskHolder waitTaskH(group, pWaitTask);
        std::thread makeTasksThread([&waitList, waitTaskH] {
          for (unsigned int i = 0; i < nTasks; ++i) {
            waitList.add(waitTaskH);
          }
        });
        std::shared_ptr<std::thread>(&makeTasksThread, join_thread);
        std::thread doneWaitThread([&waitList, waitTaskH] { waitList.doneWaiting(std::exception_ptr{}); });
        std::shared_ptr<std::thread>(&doneWaitThread, join_thread);
      }
      waitTask.wait();
    }
  }
}
