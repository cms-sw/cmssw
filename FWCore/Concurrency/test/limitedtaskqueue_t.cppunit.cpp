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
#include "tbb/task.h"
#include "FWCore/Concurrency/interface/LimitedTaskQueue.h"
#include "FWCore/Concurrency/interface/FunctorTask.h"

class LimitedTaskQueue_test : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(LimitedTaskQueue_test);
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

CPPUNIT_TEST_SUITE_REGISTRATION(LimitedTaskQueue_test);

void LimitedTaskQueue_test::testPush() {
  {
    std::atomic<unsigned int> count{0};

    edm::LimitedTaskQueue queue{1};
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

  {
    std::atomic<unsigned int> count{0};

    constexpr unsigned int kMax = 2;
    edm::LimitedTaskQueue queue{kMax};
    {
      std::shared_ptr<tbb::task> waitTask{new (tbb::task::allocate_root()) tbb::empty_task{},
                                          [](tbb::task* iTask) { tbb::task::destroy(*iTask); }};
      waitTask->set_ref_count(1 + 3);
      tbb::task* pWaitTask = waitTask.get();

      queue.push([&count, pWaitTask] {
        CPPUNIT_ASSERT(count++ < kMax);
        usleep(10);
        --count;
        pWaitTask->decrement_ref_count();
      });

      queue.push([&count, pWaitTask] {
        CPPUNIT_ASSERT(count++ < kMax);
        usleep(10);
        --count;
        pWaitTask->decrement_ref_count();
      });

      queue.push([&count, pWaitTask] {
        CPPUNIT_ASSERT(count++ < kMax);
        usleep(10);
        --count;
        pWaitTask->decrement_ref_count();
      });

      waitTask->wait_for_all();
      CPPUNIT_ASSERT(count == 0);
    }
  }
}

void LimitedTaskQueue_test::testPushAndWait() {
  {
    std::atomic<unsigned int> count{0};

    edm::LimitedTaskQueue queue{1};
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

  {
    std::atomic<unsigned int> count{0};
    std::atomic<unsigned int> countTasksRun{0};
    constexpr unsigned int kMax = 2;

    edm::LimitedTaskQueue queue{kMax};
    {
      queue.pushAndWait([&count, &countTasksRun] {
        CPPUNIT_ASSERT(count++ < kMax);
        usleep(10);
        --count;
        CPPUNIT_ASSERT(1 == ++countTasksRun);
      });

      queue.pushAndWait([&count, &countTasksRun] {
        CPPUNIT_ASSERT(count++ < kMax);
        usleep(10);
        --count;
        CPPUNIT_ASSERT(2 == ++countTasksRun);
      });

      queue.pushAndWait([&count, &countTasksRun] {
        CPPUNIT_ASSERT(count++ < kMax);
        usleep(10);
        --count;
        CPPUNIT_ASSERT(3 == ++countTasksRun);
      });

      auto c = count.load();
      if (c != 0) {
        std::cout << "ERROR count " << c << " != 0" << std::endl;
      }
      CPPUNIT_ASSERT(count == 0);

      auto v = countTasksRun.load();
      if (v != 3) {
        std::cout << "ERROR # tasks Run " << v << " != 3" << std::endl;
      }
      CPPUNIT_ASSERT(v == 3);
    }
  }
}
void LimitedTaskQueue_test::testPause() {
  std::atomic<unsigned int> count{0};

  edm::LimitedTaskQueue queue{1};
  {
    {
      std::shared_ptr<tbb::task> waitTask{new (tbb::task::allocate_root()) tbb::empty_task{},
                                          [](tbb::task* iTask) { tbb::task::destroy(*iTask); }};
      waitTask->set_ref_count(1 + 3);
      tbb::task* pWaitTask = waitTask.get();
      edm::LimitedTaskQueue::Resumer resumer;
      std::atomic<bool> resumerSet{false};
      std::exception_ptr e1;
      queue.pushAndPause([&resumer, &resumerSet, &count, pWaitTask, &e1](edm::LimitedTaskQueue::Resumer iResumer) {
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
      queue.push([&count, pWaitTask, &e2] {
        try {
          CPPUNIT_ASSERT(++count == 2);
        } catch (...) {
          e2 = std::current_exception();
        }
        pWaitTask->decrement_ref_count();
      });

      std::exception_ptr e3;
      queue.push([&count, pWaitTask, &e3] {
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
      waitTask->wait_for_all();
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
    std::shared_ptr<tbb::task> waitTask{new (tbb::task::allocate_root()) tbb::empty_task{},
                                        [](tbb::task* iTask) { tbb::task::destroy(*iTask); }};
    waitTask->set_ref_count(3);
    tbb::task* pWaitTask = waitTask.get();
    std::atomic<unsigned int> count{0};
    std::atomic<unsigned int> nRunningTasks{0};

    std::atomic<bool> waitToStart{true};
    {
      auto j = edm::make_functor_task(tbb::task::allocate_root(), [&queue, &waitToStart, pWaitTask, &count, &nRunningTasks] {
        while (waitToStart.load()) {
        };
        std::shared_ptr<tbb::task> guard{pWaitTask, [](tbb::task* iTask) { iTask->decrement_ref_count(); }};
        for (unsigned int i = 0; i < nTasks; ++i) {
          pWaitTask->increment_ref_count();
          queue.push([&count, pWaitTask, &nRunningTasks] {
            std::shared_ptr<tbb::task> guardAgain{pWaitTask, [](tbb::task* iTask) { iTask->decrement_ref_count(); }};
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
      tbb::task::enqueue(*j);

      waitToStart = false;
      for (unsigned int i = 0; i < nTasks; ++i) {
        pWaitTask->increment_ref_count();
        queue.push([&count, pWaitTask, &nRunningTasks] {
          std::shared_ptr<tbb::task> guard{pWaitTask, [](tbb::task* iTask) { iTask->decrement_ref_count(); }};
          auto nrt = nRunningTasks++;
          if (nrt >= kMax) {
            std::cout << "ERROR " << nRunningTasks << " >= " << kMax << std::endl;
          }
          CPPUNIT_ASSERT(nrt < kMax);
          ++count;
          --nRunningTasks;
        });
      }
      pWaitTask->decrement_ref_count();
    }
    waitTask->wait_for_all();

    CPPUNIT_ASSERT(0 == nRunningTasks);
    CPPUNIT_ASSERT(2 * nTasks == count);
  }
}
