//
//  LimitedTaskQueue_test.cpp
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
#include "FWCore/Concurrency/interface/LimitedTaskQueue.h"
#include "FWCore/Concurrency/interface/FunctorTask.h"

class LimitedTaskQueue_test : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(LimitedTaskQueue_test);
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

CPPUNIT_TEST_SUITE_REGISTRATION(LimitedTaskQueue_test);

void LimitedTaskQueue_test::testPush() {
  {
    std::atomic<unsigned int> count{0};

    edm::LimitedTaskQueue queue{1};
    {
      edm::FinalWaitingTask waitTask;
      auto* pWaitTask = &waitTask;
      tbb::task_group group;
      waitTask.increment_ref_count();
      waitTask.increment_ref_count();
      waitTask.increment_ref_count();

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

  {
    std::atomic<unsigned int> count{0};

    constexpr unsigned int kMax = 2;
    edm::LimitedTaskQueue queue{kMax};
    {
      edm::FinalWaitingTask waitTask;
      auto* pWaitTask = &waitTask;
      tbb::task_group group;

      waitTask.increment_ref_count();
      waitTask.increment_ref_count();
      waitTask.increment_ref_count();

      queue.push(group, [&count, pWaitTask] {
        CPPUNIT_ASSERT(count++ < kMax);
        usleep(10);
        --count;
        pWaitTask->decrement_ref_count();
      });

      queue.push(group, [&count, pWaitTask] {
        CPPUNIT_ASSERT(count++ < kMax);
        usleep(10);
        --count;
        pWaitTask->decrement_ref_count();
      });

      queue.push(group, [&count, pWaitTask] {
        CPPUNIT_ASSERT(count++ < kMax);
        usleep(10);
        --count;
        pWaitTask->decrement_ref_count();
      });

      do {
        group.wait();
      } while (not waitTask.done());
      CPPUNIT_ASSERT(count == 0);
    }
  }
}

void LimitedTaskQueue_test::testPause() {
  std::atomic<unsigned int> count{0};

  edm::LimitedTaskQueue queue{1};
  {
    {
      edm::FinalWaitingTask waitTask;
      auto* pWaitTask = &waitTask;
      tbb::task_group group;

      waitTask.increment_ref_count();
      waitTask.increment_ref_count();
      waitTask.increment_ref_count();

      edm::LimitedTaskQueue::Resumer resumer;
      std::atomic<bool> resumerSet{false};
      std::exception_ptr e1;
      queue.pushAndPause(group,
                         [&resumer, &resumerSet, &count, pWaitTask, &e1](edm::LimitedTaskQueue::Resumer iResumer) {
                           resumer = std::move(iResumer);
                           resumerSet = true;
                           try {
                             CPPUNIT_ASSERT(++count == 1);
                           } catch (...) {
                             e1 = std::current_exception();
                           }
                           pWaitTask->decrement_ref_count();
                         });

      std::exception_ptr e2;
      queue.push(group, [&count, pWaitTask, &e2] {
        try {
          CPPUNIT_ASSERT(++count == 2);
        } catch (...) {
          e2 = std::current_exception();
        }
        pWaitTask->decrement_ref_count();
      });

      std::exception_ptr e3;
      queue.push(group, [&count, pWaitTask, &e3] {
        try {
          CPPUNIT_ASSERT(++count == 3);
        } catch (...) {
          e3 = std::current_exception();
        }
        pWaitTask->decrement_ref_count();
      });
      usleep(100);
      //can't do == since the queue may not have processed the first task yet
      CPPUNIT_ASSERT(2 >= count);
      while (not resumerSet) {
      }
      CPPUNIT_ASSERT(resumer.resume());
      do {
        group.wait();
      } while (not waitTask.done());
      CPPUNIT_ASSERT(count == 3);
      if (e1) {
        std::rethrow_exception(e1);
      }
      if (e2) {
        std::rethrow_exception(e2);
      }
      if (e3) {
        std::rethrow_exception(e3);
      }
    }
  }
}

void LimitedTaskQueue_test::stressTest() {
  constexpr unsigned int kMax = 3;
  edm::LimitedTaskQueue queue{kMax};

  unsigned int index = 100;
  const unsigned int nTasks = 1000;
  while (0 != --index) {
    tbb::task_group group;
    std::atomic<int> waiting{2};
    std::atomic<unsigned int> count{0};
    std::atomic<unsigned int> nRunningTasks{0};

    std::atomic<bool> waitToStart{true};
    {
      {
        tbb::task_arena arena{tbb::task_arena::attach()};
        arena.enqueue([&queue, &waitToStart, &group, &waiting, &count, &nRunningTasks] {
          while (waitToStart.load()) {
          };
          for (unsigned int i = 0; i < nTasks; ++i) {
            ++waiting;
            queue.push(group, [&count, &waiting, &nRunningTasks] {
              std::shared_ptr<std::atomic<int>> guardAgain{&waiting, [](auto* v) { --(*v); }};
              auto nrt = nRunningTasks++;
              if (nrt >= kMax) {
                std::cout << "ERROR " << nRunningTasks << " >= " << kMax << std::endl;
              }
              CPPUNIT_ASSERT(nrt < kMax);
              ++count;
              --nRunningTasks;
            });
          }
        });
      }

      waitToStart = false;
      for (unsigned int i = 0; i < nTasks; ++i) {
        ++waiting;
        queue.push(group, [&count, &group, &waiting, &nRunningTasks] {
          std::shared_ptr<std::atomic<int>> guard{&waiting, [](auto* v) { --(*v); }};
          auto nrt = nRunningTasks++;
          if (nrt >= kMax) {
            std::cout << "ERROR " << nRunningTasks << " >= " << kMax << std::endl;
          }
          CPPUNIT_ASSERT(nrt < kMax);
          ++count;
          --nRunningTasks;
        });
      }
      --waiting;
    }
    do {
      group.wait();
    } while (0 != waiting.load());

    CPPUNIT_ASSERT(0 == nRunningTasks);
    CPPUNIT_ASSERT(2 * nTasks == count);
  }
}
