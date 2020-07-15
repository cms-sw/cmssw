//
//  SerialTaskQueue_test.cpp
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
#include <iostream>
#include "tbb/task.h"
#include "FWCore/Concurrency/interface/SerialTaskQueueChain.h"

class SerialTaskQueueChain_test : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(SerialTaskQueueChain_test);
  CPPUNIT_TEST(testPush);
  CPPUNIT_TEST(testPushOne);
  CPPUNIT_TEST(testPushAndWait);
  CPPUNIT_TEST(testPushAndWaitOne);
  CPPUNIT_TEST(stressTest);
  CPPUNIT_TEST_SUITE_END();

public:
  void testPush();
  void testPushOne();
  void testPushAndWait();
  void testPushAndWaitOne();
  void stressTest();
  void setUp() override {}
  void tearDown() override {}
};

CPPUNIT_TEST_SUITE_REGISTRATION(SerialTaskQueueChain_test);

void SerialTaskQueueChain_test::testPush() {
  std::atomic<unsigned int> count{0};

  std::vector<std::shared_ptr<edm::SerialTaskQueue>> queues = {std::make_shared<edm::SerialTaskQueue>(),
                                                               std::make_shared<edm::SerialTaskQueue>()};
  edm::SerialTaskQueueChain chain(queues);
  {
    std::shared_ptr<tbb::task> waitTask{new (tbb::task::allocate_root()) tbb::empty_task{},
                                        [](tbb::task* iTask) { tbb::task::destroy(*iTask); }};
    waitTask->set_ref_count(1 + 3);
    tbb::task* pWaitTask = waitTask.get();

    chain.push([&count, pWaitTask] {
      CPPUNIT_ASSERT(count++ == 0);
      usleep(10);
      pWaitTask->decrement_ref_count();
    });

    chain.push([&count, pWaitTask] {
      CPPUNIT_ASSERT(count++ == 1);
      usleep(10);
      pWaitTask->decrement_ref_count();
    });

    chain.push([&count, pWaitTask] {
      CPPUNIT_ASSERT(count++ == 2);
      usleep(10);
      pWaitTask->decrement_ref_count();
    });

    waitTask->wait_for_all();
    CPPUNIT_ASSERT(count == 3);
    while (chain.outstandingTasks() != 0)
      ;
  }
}

void SerialTaskQueueChain_test::testPushOne() {
  std::atomic<unsigned int> count{0};

  std::vector<std::shared_ptr<edm::SerialTaskQueue>> queues = {std::make_shared<edm::SerialTaskQueue>()};
  edm::SerialTaskQueueChain chain(queues);
  {
    std::shared_ptr<tbb::task> waitTask{new (tbb::task::allocate_root()) tbb::empty_task{},
                                        [](tbb::task* iTask) { tbb::task::destroy(*iTask); }};
    waitTask->set_ref_count(1 + 3);
    tbb::task* pWaitTask = waitTask.get();

    chain.push([&count, pWaitTask] {
      CPPUNIT_ASSERT(count++ == 0);
      usleep(10);
      pWaitTask->decrement_ref_count();
    });

    chain.push([&count, pWaitTask] {
      CPPUNIT_ASSERT(count++ == 1);
      usleep(10);
      pWaitTask->decrement_ref_count();
    });

    chain.push([&count, pWaitTask] {
      CPPUNIT_ASSERT(count++ == 2);
      usleep(10);
      pWaitTask->decrement_ref_count();
    });

    waitTask->wait_for_all();
    CPPUNIT_ASSERT(count == 3);
    while (chain.outstandingTasks() != 0)
      ;
  }
}

void SerialTaskQueueChain_test::testPushAndWait() {
  std::atomic<unsigned int> count{0};

  std::vector<std::shared_ptr<edm::SerialTaskQueue>> queues = {std::make_shared<edm::SerialTaskQueue>(),
                                                               std::make_shared<edm::SerialTaskQueue>()};
  edm::SerialTaskQueueChain chain(queues);
  {
    chain.push([&count] {
      CPPUNIT_ASSERT(count++ == 0);
      usleep(10);
    });

    chain.push([&count] {
      CPPUNIT_ASSERT(count++ == 1);
      usleep(10);
    });

    chain.pushAndWait([&count] {
      CPPUNIT_ASSERT(count++ == 2);
      usleep(10);
    });

    CPPUNIT_ASSERT(count == 3);
  }
  while (chain.outstandingTasks() != 0)
    ;
}

void SerialTaskQueueChain_test::testPushAndWaitOne() {
  std::atomic<unsigned int> count{0};

  std::vector<std::shared_ptr<edm::SerialTaskQueue>> queues = {std::make_shared<edm::SerialTaskQueue>()};
  edm::SerialTaskQueueChain chain(queues);
  {
    chain.push([&count] {
      CPPUNIT_ASSERT(count++ == 0);
      usleep(10);
    });

    chain.push([&count] {
      CPPUNIT_ASSERT(count++ == 1);
      usleep(10);
    });

    chain.pushAndWait([&count] {
      CPPUNIT_ASSERT(count++ == 2);
      usleep(10);
    });

    CPPUNIT_ASSERT(count == 3);
  }
  while (chain.outstandingTasks() != 0)
    ;
}

namespace {
  void join_thread(std::thread* iThread) {
    if (iThread->joinable()) {
      iThread->join();
    }
  }
}  // namespace

void SerialTaskQueueChain_test::stressTest() {
  std::vector<std::shared_ptr<edm::SerialTaskQueue>> queues = {std::make_shared<edm::SerialTaskQueue>(),
                                                               std::make_shared<edm::SerialTaskQueue>()};

  edm::SerialTaskQueueChain chain(queues);

  unsigned int index = 100;
  const unsigned int nTasks = 1000;
  while (0 != --index) {
    std::shared_ptr<tbb::task> waitTask{new (tbb::task::allocate_root()) tbb::empty_task{},
                                        [](tbb::task* iTask) { tbb::task::destroy(*iTask); }};
    waitTask->set_ref_count(3);
    tbb::task* pWaitTask = waitTask.get();
    std::atomic<unsigned int> count{0};

    std::atomic<bool> waitToStart{true};
    {
      std::thread pushThread([&chain, &waitToStart, pWaitTask, &count] {
        while (waitToStart.load()) {
        };
        for (unsigned int i = 0; i < nTasks; ++i) {
          pWaitTask->increment_ref_count();
          chain.push([&count, pWaitTask] {
            ++count;
            pWaitTask->decrement_ref_count();
          });
        }

        pWaitTask->decrement_ref_count();
      });

      waitToStart = false;
      for (unsigned int i = 0; i < nTasks; ++i) {
        pWaitTask->increment_ref_count();
        chain.push([&count, pWaitTask] {
          ++count;
          pWaitTask->decrement_ref_count();
        });
      }
      pWaitTask->decrement_ref_count();
      std::shared_ptr<std::thread>(&pushThread, join_thread);
    }
    waitTask->wait_for_all();

    CPPUNIT_ASSERT(2 * nTasks == count);
  }
  while (chain.outstandingTasks() != 0)
    ;
}
