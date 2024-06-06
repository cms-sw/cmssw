#include "catch.hpp"

#include "oneapi/tbb/global_control.h"

#include "FWCore/Concurrency/interface/chain_first.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingThreadPool.h"
#include "FWCore/Concurrency/interface/hardware_pause.h"

namespace {
  constexpr char const* errorContext() { return "WaitingThreadPool test"; }
}  // namespace

TEST_CASE("Test WaitingThreadPool", "[edm::WaitingThreadPool") {
  // Using parallelism 2 here because otherwise the
  // tbb::task_arena::enqueue() in WaitingTaskWithArenaHolder will
  // start a new TBB thread that "inherits" the name from the
  // WaitingThreadPool thread.
  oneapi::tbb::global_control control(oneapi::tbb::global_control::max_allowed_parallelism, 2);
  edm::WaitingThreadPool pool;

  SECTION("One async call") {
    std::atomic<int> count{0};

    oneapi::tbb::task_group group;
    edm::FinalWaitingTask waitTask{group};
    {
      using namespace edm::waiting_task::chain;
      auto h = first([&pool, &count](edm::WaitingTaskHolder h) {
                 edm::WaitingTaskWithArenaHolder h2(std::move(h));
                 pool.runAsync(
                     std::move(h2), [&count]() { ++count; }, errorContext);
               }) |
               lastTask(edm::WaitingTaskHolder(group, &waitTask));
      h.doneWaiting(std::exception_ptr());
    }
    waitTask.waitNoThrow();
    REQUIRE(count.load() == 1);
    REQUIRE(waitTask.done());
    REQUIRE(not waitTask.exceptionPtr());
  }

  SECTION("Two async calls") {
    std::atomic<int> count{0};
    std::atomic<bool> mayContinue{false};

    oneapi::tbb::task_group group;
    edm::FinalWaitingTask waitTask{group};

    {
      using namespace edm::waiting_task::chain;
      auto h = first([&pool, &count, &mayContinue](edm::WaitingTaskHolder h) {
                 edm::WaitingTaskWithArenaHolder h2(std::move(h));
                 pool.runAsync(
                     h2,
                     [&count, &mayContinue]() {
                       while (not mayContinue) {
                         hardware_pause();
                       }
                       using namespace std::chrono_literals;
                       std::this_thread::sleep_for(10ms);
                       ++count;
                     },
                     errorContext);
                 pool.runAsync(
                     h2, [&count]() { ++count; }, errorContext);
               }) |
               lastTask(edm::WaitingTaskHolder(group, &waitTask));
      h.doneWaiting(std::exception_ptr());
    }
    mayContinue = true;
    waitTask.waitNoThrow();
    REQUIRE(count.load() == 2);
    REQUIRE(waitTask.done());
    REQUIRE(not waitTask.exceptionPtr());
  }

  SECTION("Concurrent async calls") {
    std::atomic<int> count{0};
    std::atomic<int> mayContinue{0};

    oneapi::tbb::task_group group;
    edm::FinalWaitingTask waitTask{group};

    {
      using namespace edm::waiting_task::chain;
      auto h1 = first([&pool, &count, &mayContinue](edm::WaitingTaskHolder h) {
                  edm::WaitingTaskWithArenaHolder h2(std::move(h));
                  ++mayContinue;
                  while (mayContinue != 2) {
                    hardware_pause();
                  }
                  pool.runAsync(
                      h2, [&count]() { ++count; }, errorContext);
                }) |
                lastTask(edm::WaitingTaskHolder(group, &waitTask));

      auto h2 = first([&pool, &count, &mayContinue](edm::WaitingTaskHolder h) {
                  edm::WaitingTaskWithArenaHolder h2(std::move(h));
                  ++mayContinue;
                  while (mayContinue != 2) {
                    hardware_pause();
                  }
                  pool.runAsync(
                      h2, [&count]() { ++count; }, errorContext);
                }) |
                lastTask(edm::WaitingTaskHolder(group, &waitTask));
      h2.doneWaiting(std::exception_ptr());
      h1.doneWaiting(std::exception_ptr());
    }
    waitTask.waitNoThrow();
    REQUIRE(count.load() == 2);
    REQUIRE(waitTask.done());
    REQUIRE(not waitTask.exceptionPtr());
  }

  SECTION("Exceptions") {
    SECTION("One async call") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};

      {
        using namespace edm::waiting_task::chain;
        auto h = first([&pool, &count](edm::WaitingTaskHolder h) {
                   edm::WaitingTaskWithArenaHolder h2(std::move(h));
                   pool.runAsync(
                       std::move(h2), [&count]() { throw std::runtime_error("error"); }, errorContext);
                 }) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));
        h.doneWaiting(std::exception_ptr());
      }
      REQUIRE_THROWS_WITH(
          waitTask.wait(),
          Catch::Contains("error") and Catch::Contains("StdException") and Catch::Contains("WaitingThreadPool test"));
      REQUIRE(count.load() == 0);
    }

    SECTION("Two async calls") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};

      {
        using namespace edm::waiting_task::chain;
        auto h = first([&pool, &count](edm::WaitingTaskHolder h) {
                   edm::WaitingTaskWithArenaHolder h2(std::move(h));
                   pool.runAsync(
                       h2,
                       [&count]() {
                         if (count.fetch_add(1) == 0) {
                           throw cms::Exception("error 1");
                         }
                         ++count;
                       },
                       errorContext);
                   pool.runAsync(
                       h2,
                       [&count]() {
                         if (count.fetch_add(1) == 0) {
                           throw cms::Exception("error 2");
                         }
                         ++count;
                       },
                       errorContext);
                 }) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));
        h.doneWaiting(std::exception_ptr());
      }
      REQUIRE_THROWS_AS(waitTask.wait(), cms::Exception);
      REQUIRE(count.load() == 3);
    }

    SECTION("Two exceptions") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};

      {
        using namespace edm::waiting_task::chain;
        auto h = first([&pool, &count](edm::WaitingTaskHolder h) {
                   edm::WaitingTaskWithArenaHolder h2(std::move(h));
                   pool.runAsync(
                       h2,
                       [&count]() {
                         ++count;
                         throw cms::Exception("error 1");
                       },
                       errorContext);
                   pool.runAsync(
                       h2,
                       [&count]() {
                         ++count;
                         throw cms::Exception("error 2");
                       },
                       errorContext);
                 }) |
                 lastTask(edm::WaitingTaskHolder(group, &waitTask));
        h.doneWaiting(std::exception_ptr());
      }
      REQUIRE_THROWS_AS(waitTask.wait(), cms::Exception);
      REQUIRE(count.load() == 2);
    }

    SECTION("Concurrent exceptions") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};

      {
        using namespace edm::waiting_task::chain;
        auto h1 = first([&pool, &count](edm::WaitingTaskHolder h) {
                    edm::WaitingTaskWithArenaHolder h2(std::move(h));
                    pool.runAsync(
                        h2,
                        [&count]() {
                          ++count;
                          throw cms::Exception("error 1");
                        },
                        errorContext);
                  }) |
                  lastTask(edm::WaitingTaskHolder(group, &waitTask));

        auto h2 = first([&pool, &count](edm::WaitingTaskHolder h) {
                    edm::WaitingTaskWithArenaHolder h2(std::move(h));
                    pool.runAsync(
                        h2,
                        [&count]() {
                          ++count;
                          throw cms::Exception("error 2");
                        },
                        errorContext);
                  }) |
                  lastTask(edm::WaitingTaskHolder(group, &waitTask));
        h2.doneWaiting(std::exception_ptr());
        h1.doneWaiting(std::exception_ptr());
      }
      REQUIRE_THROWS_AS(waitTask.wait(), cms::Exception);
      REQUIRE(count.load() == 2);
    }

    SECTION("Concurrent exception and success") {
      std::atomic<int> count{0};

      oneapi::tbb::task_group group;
      edm::FinalWaitingTask waitTask{group};

      {
        using namespace edm::waiting_task::chain;
        auto h1 = first([&pool, &count](edm::WaitingTaskHolder h) {
                    edm::WaitingTaskWithArenaHolder h2(std::move(h));
                    pool.runAsync(
                        h2,
                        [&count]() {
                          ++count;
                          throw cms::Exception("error 1");
                        },
                        errorContext);
                  }) |
                  lastTask(edm::WaitingTaskHolder(group, &waitTask));

        auto h2 = first([&pool, &count](edm::WaitingTaskHolder h) {
                    edm::WaitingTaskWithArenaHolder h2(std::move(h));
                    pool.runAsync(
                        h2, [&count]() { ++count; }, errorContext);
                  }) |
                  lastTask(edm::WaitingTaskHolder(group, &waitTask));
        h2.doneWaiting(std::exception_ptr());
        h1.doneWaiting(std::exception_ptr());
      }
      REQUIRE_THROWS_AS(waitTask.wait(), cms::Exception);
      REQUIRE(count.load() == 2);
    }
  }
}
