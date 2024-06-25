#include "catch.hpp"

#include <atomic>

#include "oneapi/tbb/global_control.h"

#include "FWCore/Concurrency/interface/chain_first.h"
#include "FWCore/Concurrency/interface/FinalWaitingTask.h"
#include "FWCore/Concurrency/interface/Async.h"

namespace {
  constexpr char const* errorContext() { return "AsyncServiceTest"; }

  class AsyncServiceTest : public edm::Async {
  public:
    enum class State { kAllowed, kDisallowed, kShutdown };

    AsyncServiceTest() = default;

    void setAllowed(bool allowed) noexcept { allowed_ = allowed; }

  private:
    void ensureAllowed() const final {
      if (not allowed_) {
        throw std::runtime_error("Calling run in this context is not allowed");
      }
    }

    std::atomic<bool> allowed_ = true;
  };
}  // namespace

TEST_CASE("Test Async", "[edm::Async") {
  // Using parallelism 2 here because otherwise the
  // tbb::task_arena::enqueue() in WaitingTaskWithArenaHolder will
  // start a new TBB thread that "inherits" the name from the
  // WaitingThreadPool thread.
  oneapi::tbb::global_control control(oneapi::tbb::global_control::max_allowed_parallelism, 2);

  SECTION("Normal operation") {
    AsyncServiceTest service;
    std::atomic<int> count{0};

    oneapi::tbb::task_group group;
    edm::FinalWaitingTask waitTask{group};

    {
      using namespace edm::waiting_task::chain;
      auto h1 = first([&service, &count](edm::WaitingTaskHolder h) {
                  edm::WaitingTaskWithArenaHolder h2(std::move(h));
                  service.runAsync(
                      h2, [&count]() { ++count; }, errorContext);
                }) |
                lastTask(edm::WaitingTaskHolder(group, &waitTask));

      auto h2 = first([&service, &count](edm::WaitingTaskHolder h) {
                  edm::WaitingTaskWithArenaHolder h2(std::move(h));
                  service.runAsync(
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

  SECTION("Disallowed") {
    AsyncServiceTest service;
    std::atomic<int> count{0};

    oneapi::tbb::task_group group;
    edm::FinalWaitingTask waitTask{group};

    {
      using namespace edm::waiting_task::chain;
      auto h = first([&service, &count](edm::WaitingTaskHolder h) {
                 edm::WaitingTaskWithArenaHolder h2(std::move(h));
                 service.runAsync(
                     h2, [&count]() { ++count; }, errorContext);
                 service.setAllowed(false);
               }) |
               then([&service, &count](edm::WaitingTaskHolder h) {
                 edm::WaitingTaskWithArenaHolder h2(std::move(h));
                 service.runAsync(
                     h2, [&count]() { ++count; }, errorContext);
               }) |
               lastTask(edm::WaitingTaskHolder(group, &waitTask));
      h.doneWaiting(std::exception_ptr());
    }
    waitTask.waitNoThrow();
    REQUIRE(count.load() == 1);
    REQUIRE(waitTask.done());
    REQUIRE(waitTask.exceptionPtr());
    REQUIRE_THROWS_WITH(std::rethrow_exception(waitTask.exceptionPtr()),
                        Catch::Contains("Calling run in this context is not allowed"));
  }
}
