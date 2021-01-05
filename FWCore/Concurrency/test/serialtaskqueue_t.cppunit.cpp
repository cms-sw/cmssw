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
#include "tbb/task_arena.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"
#include "FWCore/Concurrency/interface/FunctorTask.h"

class SerialTaskQueue_test : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(SerialTaskQueue_test);
  CPPUNIT_TEST(testPush);
  CPPUNIT_TEST(testPause);
  CPPUNIT_TEST(stressTest);
  CPPUNIT_TEST_SUITE_END();

public:
  void testPush();
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
    edm::FinalWaitingTask waitTask;
    auto* pWaitTask = &waitTask;
    tbb::task_group group;

    queue.push(group, [&count, pWaitTask] {
      CPPUNIT_ASSERT(count++ == 0);
      usleep(10);
      pWaitTask->decrement_ref_count();
    });

    queue.push(group, [&count, pWaitTask] {
      CPPUNIT_ASSERT(count++ == 1);
      usleep(10);
      pWaitTask->decrement_ref_count();
    });

    queue.push(group, [&count, pWaitTask] {
      CPPUNIT_ASSERT(count++ == 2);
      usleep(10);
      pWaitTask->decrement_ref_count();
    });

    do {
      group.wait();
    } while (not waitTask.done());
    CPPUNIT_ASSERT(count == 3);
  }
}

void SerialTaskQueue_test::testPause() {
  std::atomic<unsigned int> count{0};

  edm::SerialTaskQueue queue;
  {
    queue.pause();
    {
      edm::FinalWaitingTask waitTask;
      auto* pWaitTask = &waitTask;
      tbb::task_group group;

      waitTask.increment_ref_count();

      queue.push(group, [&count, pWaitTask] {
        CPPUNIT_ASSERT(count++ == 0);
        pWaitTask->decrement_ref_count();
      });
      usleep(1000);
      CPPUNIT_ASSERT(0 == count);
      queue.resume();
      do {
        group.wait();
      } while (not waitTask.done());
      CPPUNIT_ASSERT(count == 1);
    }

    {
      edm::FinalWaitingTask waitTask;
      auto* pWaitTask = &waitTask;
      tbb::task_group group;

      waitTask.increment_ref_count();
      waitTask.increment_ref_count();
      waitTask.increment_ref_count();

      queue.push(group, [&count, &queue, pWaitTask] {
        queue.pause();
        CPPUNIT_ASSERT(count++ == 1);
        pWaitTask->decrement_ref_count();
      });
      queue.push(group, [&count, pWaitTask] {
        CPPUNIT_ASSERT(count++ == 2);
        pWaitTask->decrement_ref_count();
      });
      queue.push(group, [&count, pWaitTask] {
        CPPUNIT_ASSERT(count++ == 3);
        pWaitTask->decrement_ref_count();
      });
      usleep(100);
      //can't do == since the queue may not have processed the first task yet
      CPPUNIT_ASSERT(2 >= count);
      queue.resume();
      do {
        group.wait();
      } while (not waitTask.done());
      CPPUNIT_ASSERT(count == 4);
    }
  }
}

void SerialTaskQueue_test::stressTest() {
  edm::SerialTaskQueue queue;

  unsigned int index = 100;
  const unsigned int nTasks = 1000;
  while (0 != --index) {
    edm::FinalWaitingTask waitTask;
    auto* pWaitTask = &waitTask;
    tbb::task_group group;
    std::atomic<unsigned int> count{0};

    waitTask.increment_ref_count();
    waitTask.increment_ref_count();

    std::atomic<bool> waitToStart{true};
    {
      {
        tbb::task_arena arena{tbb::task_arena::attach()};
        arena.enqueue([&queue, &waitToStart, pWaitTask, &count, &group] {
          while (waitToStart.load()) {
          };
          for (unsigned int i = 0; i < nTasks; ++i) {
            pWaitTask->increment_ref_count();
            queue.push(group, [&count, pWaitTask] {
              ++count;
              pWaitTask->decrement_ref_count();
            });
          }

          pWaitTask->decrement_ref_count();
        });
      }

      waitToStart = false;
      for (unsigned int i = 0; i < nTasks; ++i) {
        pWaitTask->increment_ref_count();
        queue.push(group, [&count, pWaitTask] {
          ++count;
          pWaitTask->decrement_ref_count();
        });
      }
      pWaitTask->decrement_ref_count();
    }
    do {
      group.wait();
    } while (not waitTask.done());

    CPPUNIT_ASSERT(2 * nTasks == count);
  }
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
