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
#include "tbb/task.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"
#include "FWCore/Concurrency/interface/FunctorTask.h"

class SerialTaskQueue_test : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(SerialTaskQueue_test);
  CPPUNIT_TEST(testPush);
  CPPUNIT_TEST(testPushAndWait);
  CPPUNIT_TEST(testPause);
  CPPUNIT_TEST(stressTest);
  CPPUNIT_TEST_SUITE_END();

public:
  void testPush();
  void testPushAndWait();
  void testPause();
  void stressTest();
  void setUp() {}
  void tearDown() {}
};

CPPUNIT_TEST_SUITE_REGISTRATION(SerialTaskQueue_test);

void SerialTaskQueue_test::testPush() {
  std::atomic<unsigned int> count{0};

  edm::SerialTaskQueue queue;
  {
    std::shared_ptr<tbb::task> waitTask{new (tbb::task::allocate_root()) tbb::empty_task{},
                                        [](tbb::task* iTask) { tbb::task::destroy(*iTask); }};
    waitTask->set_ref_count(1 + 3);
    tbb::task* pWaitTask = waitTask.get();

    queue.push([&count, pWaitTask] {
      CPPUNIT_ASSERT(count++ == 0);
      usleep(10);
      pWaitTask->decrement_ref_count();
    });

    queue.push([&count, pWaitTask] {
      CPPUNIT_ASSERT(count++ == 1);
      usleep(10);
      pWaitTask->decrement_ref_count();
    });

    queue.push([&count, pWaitTask] {
      CPPUNIT_ASSERT(count++ == 2);
      usleep(10);
      pWaitTask->decrement_ref_count();
    });

    waitTask->wait_for_all();
    CPPUNIT_ASSERT(count == 3);
  }
}

void SerialTaskQueue_test::testPushAndWait() {
  std::atomic<unsigned int> count{0};

  edm::SerialTaskQueue queue;
  {
    queue.push([&count] {
      CPPUNIT_ASSERT(count++ == 0);
      usleep(10);
    });

    queue.push([&count] {
      CPPUNIT_ASSERT(count++ == 1);
      usleep(10);
    });

    queue.pushAndWait([&count] {
      CPPUNIT_ASSERT(count++ == 2);
      usleep(10);
    });

    CPPUNIT_ASSERT(count == 3);
  }
}

void SerialTaskQueue_test::testPause() {
  std::atomic<unsigned int> count{0};

  edm::SerialTaskQueue queue;
  {
    queue.pause();
    {
      std::shared_ptr<tbb::task> waitTask{new (tbb::task::allocate_root()) tbb::empty_task{},
                                          [](tbb::task* iTask) { tbb::task::destroy(*iTask); }};
      waitTask->set_ref_count(1 + 1);
      tbb::task* pWaitTask = waitTask.get();

      queue.push([&count, pWaitTask] {
        CPPUNIT_ASSERT(count++ == 0);
        pWaitTask->decrement_ref_count();
      });
      usleep(1000);
      CPPUNIT_ASSERT(0 == count);
      queue.resume();
      waitTask->wait_for_all();
      CPPUNIT_ASSERT(count == 1);
    }

    {
      std::shared_ptr<tbb::task> waitTask{new (tbb::task::allocate_root()) tbb::empty_task{},
                                          [](tbb::task* iTask) { tbb::task::destroy(*iTask); }};
      waitTask->set_ref_count(1 + 3);
      tbb::task* pWaitTask = waitTask.get();

      queue.push([&count, &queue, pWaitTask] {
        queue.pause();
        CPPUNIT_ASSERT(count++ == 1);
        pWaitTask->decrement_ref_count();
      });
      queue.push([&count, pWaitTask] {
        CPPUNIT_ASSERT(count++ == 2);
        pWaitTask->decrement_ref_count();
      });
      queue.push([&count, pWaitTask] {
        CPPUNIT_ASSERT(count++ == 3);
        pWaitTask->decrement_ref_count();
      });
      usleep(100);
      //can't do == since the queue may not have processed the first task yet
      CPPUNIT_ASSERT(2 >= count);
      queue.resume();
      waitTask->wait_for_all();
      CPPUNIT_ASSERT(count == 4);
    }
  }
}

void SerialTaskQueue_test::stressTest() {
  edm::SerialTaskQueue queue;

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
      auto j = edm::make_functor_task(tbb::task::allocate_root(), [&queue, &waitToStart, pWaitTask, &count] {
        while (waitToStart.load()) {
        };
        for (unsigned int i = 0; i < nTasks; ++i) {
          pWaitTask->increment_ref_count();
          queue.push([&count, pWaitTask] {
            ++count;
            pWaitTask->decrement_ref_count();
          });
        }

        pWaitTask->decrement_ref_count();
      });
      tbb::task::enqueue(*j);

      waitToStart = false;
      for (unsigned int i = 0; i < nTasks; ++i) {
        pWaitTask->increment_ref_count();
        queue.push([&count, pWaitTask] {
          ++count;
          pWaitTask->decrement_ref_count();
        });
      }
      pWaitTask->decrement_ref_count();
    }
    waitTask->wait_for_all();

    CPPUNIT_ASSERT(2 * nTasks == count);
  }
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
