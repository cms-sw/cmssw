#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"

namespace cms::perftools {
  class AllocTester {
  public:
    void callAlloc(size_t iRequested, size_t iActual) {
      reg_.allocCalled(
          iRequested, []() { return nullptr; }, [iActual](auto) { return iActual; });
    }

    void callDealloc(size_t iActual) {
      reg_.deallocCalled(
          reinterpret_cast<void*>(1), [](auto) {}, [iActual](auto) { return iActual; });
    }

    template <typename A>
    void callAlloc(size_t iRequested, size_t iActual, A&& iAlloc) {
      reg_.allocCalled(iRequested, std::forward<A>(iAlloc), [iActual](auto) { return iActual; });
    }

    template <typename D>
    void callDealloc(size_t iActual, D&& iDealloc) {
      reg_.deallocCalled(reinterpret_cast<void*>(1), std::forward<D>(iDealloc), [iActual](auto) { return iActual; });
    }

    void callDeallocNull() {
      reg_.deallocCalled(
          nullptr, [](auto) {}, [](auto) { return 0; });
    }

    AllocMonitorRegistry reg_;
  };
}  // namespace cms::perftools

using namespace cms::perftools;

namespace {
  int s_calls = 0;

  class TestCallMonitor : public AllocMonitorBase {
  public:
    TestCallMonitor(int) { ++s_calls; }

    ~TestCallMonitor() override { ++s_calls; }

    void allocCalled(size_t iRequestedSize, size_t iActualSize) final { ++s_calls; }
    void deallocCalled(size_t iActualSize) final { ++s_calls; }
  };

  bool s_started = false;
  bool s_stopped = false;

  class TestRecursionMonitor : public AllocMonitorBase {
  public:
    TestRecursionMonitor(AllocTester* iTester) : tester_(iTester) {
      ++s_calls;
      tester_->callAlloc(1, 1);
    }

    ~TestRecursionMonitor() override {
      ++s_calls;
      tester_->callDealloc(1);
    }

    void allocCalled(size_t iRequestedSize, size_t iActualSize) final {
      ++s_calls;
      tester_->callAlloc(1, 1);
      tester_->callDealloc(1);
    }
    void deallocCalled(size_t iActualSize) final {
      ++s_calls;
      tester_->callAlloc(1, 1);
      tester_->callDealloc(1);
    }

  private:
    AllocTester* tester_;
  };
}  // namespace

extern "C" {
void alloc_monitor_start() { s_started = true; }
void alloc_monitor_stop() { s_stopped = true; }
}

TEST_CASE("Test API for AllocMonitorRegistry", "[AllocMonitorRegistry]") {
  SECTION("Calls Check") {
    CHECK(0 == s_calls);
    CHECK(s_started == false);
    CHECK(s_stopped == false);
    {
      AllocTester t;
      CHECK(s_started == false);
      CHECK(s_stopped == false);

      auto tester = t.reg_.createAndRegisterMonitor<TestCallMonitor>(1);
      CHECK(s_started == true);
      CHECK(s_stopped == false);
      CHECK(1 == s_calls);
      CHECK(tester != nullptr);

      t.callAlloc(1, 1);
      CHECK(2 == s_calls);

      t.callDealloc(1);
      CHECK(3 == s_calls);

      t.reg_.deregisterMonitor(tester);
      CHECK(4 == s_calls);
    }
    CHECK(4 == s_calls);
    CHECK(s_stopped == true);
    s_started = false;
    s_stopped = false;
    s_calls = 0;
  }
  SECTION("Null delete") {
    {
      AllocTester t;
      CHECK(s_started == false);
      CHECK(s_stopped == false);

      auto tester = t.reg_.createAndRegisterMonitor<TestCallMonitor>(1);
      CHECK(s_started == true);
      CHECK(s_stopped == false);
      CHECK(1 == s_calls);
      CHECK(tester != nullptr);

      t.callDeallocNull();
      CHECK(1 == s_calls);
      t.reg_.deregisterMonitor(tester);
      CHECK(2 == s_calls);
    }
    s_started = false;
    s_stopped = false;
    s_calls = 0;
  }
  SECTION("Recursion in monitor") {
    CHECK(0 == s_calls);
    CHECK(s_started == false);
    CHECK(s_stopped == false);
    {
      AllocTester t;
      CHECK(s_started == false);
      CHECK(s_stopped == false);

      auto tester = t.reg_.createAndRegisterMonitor<TestRecursionMonitor>(&t);
      CHECK(s_started == true);
      CHECK(s_stopped == false);
      CHECK(1 == s_calls);
      CHECK(tester != nullptr);

      t.callAlloc(1, 1);
      CHECK(2 == s_calls);

      t.callDealloc(1);
      CHECK(3 == s_calls);

      t.reg_.deregisterMonitor(tester);
      CHECK(4 == s_calls);
    }
    CHECK(4 == s_calls);
    CHECK(s_stopped == true);
    s_started = false;
    s_stopped = false;
    s_calls = 0;
  }
  SECTION("System calling system") {
    CHECK(0 == s_calls);
    CHECK(s_started == false);
    CHECK(s_stopped == false);
    {
      AllocTester t;
      CHECK(s_started == false);
      CHECK(s_stopped == false);

      auto tester = t.reg_.createAndRegisterMonitor<TestCallMonitor>(1);
      CHECK(s_started == true);
      CHECK(s_stopped == false);
      CHECK(1 == s_calls);
      CHECK(tester != nullptr);

      t.callAlloc(1, 1, [&t]() {
        t.callAlloc(1, 1);
        return 1;
      });
      CHECK(2 == s_calls);

      t.callDealloc(1, [&t](auto) { t.callDealloc(1); });
      CHECK(3 == s_calls);

      t.reg_.deregisterMonitor(tester);
      CHECK(4 == s_calls);
    }
    CHECK(4 == s_calls);
    CHECK(s_stopped == true);
    s_started = false;
    s_stopped = false;
    s_calls = 0;
  }
}
