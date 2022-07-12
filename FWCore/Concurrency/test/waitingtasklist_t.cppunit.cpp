//
//  WaitingTaskList_test.cpp
//  DispatchProcessingDemo
//
//  Created by Chris Jones on 9/27/11.
//

#include <iostream>

#include <cppunit/extensions/HelperMacros.h>
#include <unistd.h>
#include <memory>
#include <atomic>
#include <thread>
#include "oneapi/tbb/task.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"

class WaitingTaskList_test : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(WaitingTaskList_test);
  CPPUNIT_TEST(addThenDone);
  CPPUNIT_TEST(doneThenAdd);
  CPPUNIT_TEST(addThenDoneFailed);
  CPPUNIT_TEST(doneThenAddFailed);
  CPPUNIT_TEST(stressTest);
  CPPUNIT_TEST_SUITE_END();

public:
  void addThenDone();
  void doneThenAdd();
  void addThenDoneFailed();
  void doneThenAddFailed();
  void stressTest();
  void setUp() {}
  void tearDown() {}
};

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
      CPPUNIT_ASSERT(m_value);
      return;
    }

  private:
    std::atomic<bool>& m_value;
  };

}  // namespace

void WaitingTaskList_test::addThenDone() {
  std::atomic<bool> called{false};

  edm::WaitingTaskList waitList;
  {
    std::exception_ptr excPtr;

    auto t = new TestCalledTask{called, excPtr};

    oneapi::tbb::task_group group;
    waitList.add(edm::WaitingTaskHolder(group, t));

    usleep(10);
    CPPUNIT_ASSERT(false == called);

    waitList.doneWaiting(std::exception_ptr{});
    group.wait();
    CPPUNIT_ASSERT(true == called);
    CPPUNIT_ASSERT(bool(excPtr) == false);
  }

  waitList.reset();
  called = false;

  {
    std::exception_ptr excPtr;

    auto t = new TestCalledTask{called, excPtr};

    oneapi::tbb::task_group group;

    waitList.add(edm::WaitingTaskHolder(group, t));

    usleep(10);
    CPPUNIT_ASSERT(false == called);

    waitList.doneWaiting(std::exception_ptr{});
    group.wait();
    CPPUNIT_ASSERT(true == called);
    CPPUNIT_ASSERT(bool(excPtr) == false);
  }
}

void WaitingTaskList_test::doneThenAdd() {
  std::atomic<bool> called{false};
  std::exception_ptr excPtr;

  edm::WaitingTaskList waitList;
  {
    oneapi::tbb::task_group group;

    auto t = new TestCalledTask{called, excPtr};

    waitList.doneWaiting(std::exception_ptr{});

    waitList.add(edm::WaitingTaskHolder(group, t));
    group.wait();
    CPPUNIT_ASSERT(true == called);
    CPPUNIT_ASSERT(bool(excPtr) == false);
  }
}

void WaitingTaskList_test::addThenDoneFailed() {
  std::atomic<bool> called{false};

  edm::WaitingTaskList waitList;
  {
    std::exception_ptr excPtr;

    auto t = new TestCalledTask{called, excPtr};

    oneapi::tbb::task_group group;

    waitList.add(edm::WaitingTaskHolder(group, t));

    usleep(10);
    CPPUNIT_ASSERT(false == called);

    waitList.doneWaiting(std::make_exception_ptr(std::string("failed")));
    group.wait();
    CPPUNIT_ASSERT(true == called);
    CPPUNIT_ASSERT(bool(excPtr) == true);
  }
}

void WaitingTaskList_test::doneThenAddFailed() {
  std::atomic<bool> called{false};
  std::exception_ptr excPtr;

  edm::WaitingTaskList waitList;
  {
    auto t = new TestCalledTask{called, excPtr};

    waitList.doneWaiting(std::make_exception_ptr(std::string("failed")));

    oneapi::tbb::task_group group;

    waitList.add(edm::WaitingTaskHolder(group, t));
    group.wait();
    CPPUNIT_ASSERT(true == called);
    CPPUNIT_ASSERT(bool(excPtr) == true);
  }
}

namespace {
  void join_thread(std::thread* iThread) {
    if (iThread->joinable()) {
      iThread->join();
    }
  }
}  // namespace

void WaitingTaskList_test::stressTest() {
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

CPPUNIT_TEST_SUITE_REGISTRATION(WaitingTaskList_test);
