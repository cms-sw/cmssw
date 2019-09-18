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
#include "tbb/task.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"

#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7)
#define CXX_THREAD_AVAILABLE
#endif

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

    tbb::task* execute() {
      if (exceptionPtr()) {
        m_ptr = *exceptionPtr();
      }
      m_called = true;
      return nullptr;
    }

  private:
    std::atomic<bool>& m_called;
    std::exception_ptr& m_ptr;
  };

  class TestValueSetTask : public edm::WaitingTask {
  public:
    TestValueSetTask(std::atomic<bool>& iValue) : m_value(iValue) {}
    tbb::task* execute() {
      CPPUNIT_ASSERT(m_value);
      return nullptr;
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

    auto waitTask = edm::make_empty_waiting_task();
    waitTask->set_ref_count(2);
    //NOTE: allocate_child does NOT increment the ref_count of waitTask!
    auto t = new (waitTask->allocate_child()) TestCalledTask{called, excPtr};

    waitList.add(t);

    usleep(10);
    __sync_synchronize();
    CPPUNIT_ASSERT(false == called);

    waitList.doneWaiting(std::exception_ptr{});
    waitTask->wait_for_all();
    __sync_synchronize();
    CPPUNIT_ASSERT(true == called);
    CPPUNIT_ASSERT(bool(excPtr) == false);
  }

  waitList.reset();
  called = false;

  {
    std::exception_ptr excPtr;

    auto waitTask = edm::make_empty_waiting_task();
    waitTask->set_ref_count(2);

    auto t = new (waitTask->allocate_child()) TestCalledTask{called, excPtr};

    waitList.add(t);

    usleep(10);
    CPPUNIT_ASSERT(false == called);

    waitList.doneWaiting(std::exception_ptr{});
    waitTask->wait_for_all();
    CPPUNIT_ASSERT(true == called);
    CPPUNIT_ASSERT(bool(excPtr) == false);
  }
}

void WaitingTaskList_test::doneThenAdd() {
  std::atomic<bool> called{false};
  std::exception_ptr excPtr;

  edm::WaitingTaskList waitList;
  {
    auto waitTask = edm::make_empty_waiting_task();
    waitTask->set_ref_count(2);

    auto t = new (waitTask->allocate_child()) TestCalledTask{called, excPtr};

    waitList.doneWaiting(std::exception_ptr{});

    waitList.add(t);
    waitTask->wait_for_all();
    CPPUNIT_ASSERT(true == called);
    CPPUNIT_ASSERT(bool(excPtr) == false);
  }
}

void WaitingTaskList_test::addThenDoneFailed() {
  std::atomic<bool> called{false};

  edm::WaitingTaskList waitList;
  {
    std::exception_ptr excPtr;

    auto waitTask = edm::make_empty_waiting_task();
    waitTask->set_ref_count(2);

    auto t = new (waitTask->allocate_child()) TestCalledTask{called, excPtr};

    waitList.add(t);

    usleep(10);
    CPPUNIT_ASSERT(false == called);

    waitList.doneWaiting(std::make_exception_ptr(std::string("failed")));
    waitTask->wait_for_all();
    CPPUNIT_ASSERT(true == called);
    CPPUNIT_ASSERT(bool(excPtr) == true);
  }
}

void WaitingTaskList_test::doneThenAddFailed() {
  std::atomic<bool> called{false};
  std::exception_ptr excPtr;

  edm::WaitingTaskList waitList;
  {
    auto waitTask = edm::make_empty_waiting_task();
    waitTask->set_ref_count(2);

    auto t = new (waitTask->allocate_child()) TestCalledTask{called, excPtr};

    waitList.doneWaiting(std::make_exception_ptr(std::string("failed")));

    waitList.add(t);
    waitTask->wait_for_all();
    CPPUNIT_ASSERT(true == called);
    CPPUNIT_ASSERT(bool(excPtr) == true);
  }
}

namespace {
#if defined(CXX_THREAD_AVAILABLE)
  void join_thread(std::thread* iThread) {
    if (iThread->joinable()) {
      iThread->join();
    }
  }
#endif
}  // namespace

void WaitingTaskList_test::stressTest() {
#if defined(CXX_THREAD_AVAILABLE)
  std::atomic<bool> called{false};
  std::exception_ptr excPtr;
  edm::WaitingTaskList waitList;

  unsigned int index = 1000;
  const unsigned int nTasks = 10000;
  while (0 != --index) {
    called = false;
    auto waitTask = edm::make_empty_waiting_task();
    waitTask->set_ref_count(3);
    tbb::task* pWaitTask = waitTask.get();

    {
      std::thread makeTasksThread([&waitList, pWaitTask, &called, &excPtr] {
        for (unsigned int i = 0; i < nTasks; ++i) {
          auto t = new (tbb::task::allocate_additional_child_of(*pWaitTask)) TestCalledTask{called, excPtr};
          waitList.add(t);
        }

        pWaitTask->decrement_ref_count();
      });
      std::shared_ptr<std::thread>(&makeTasksThread, join_thread);

      std::thread doneWaitThread([&waitList, &called, pWaitTask] {
        called = true;
        waitList.doneWaiting(std::exception_ptr{});
        pWaitTask->decrement_ref_count();
      });
      std::shared_ptr<std::thread>(&doneWaitThread, join_thread);
    }
    waitTask->wait_for_all();
  }
#endif
}

CPPUNIT_TEST_SUITE_REGISTRATION(WaitingTaskList_test);
