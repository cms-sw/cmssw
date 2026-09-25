#include "FWCore/Concurrency/interface/SpinLock.h"
#include <catch2/catch_all.hpp>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>

TEST_CASE("SpinLock", "[SpinLock]") {
  SECTION("construct/destruct") { edm::SpinLock lock; }
  SECTION("lock guard") {
    edm::SpinLock lock;
    std::lock_guard<edm::SpinLock> guard{lock};
  }
  SECTION("thread") {
    edm::SpinLock lock;

    std::atomic<int> wait_count{2};
    std::atomic<int> concurrent_count{0};

    std::exception_ptr excpt;
    std::thread t{[&lock, &wait_count, &concurrent_count, &excpt]() {
      --wait_count;
      while (0 != wait_count) {
      }
      try {
        std::lock_guard<edm::SpinLock> g{lock};
        auto v = ++concurrent_count;
        REQUIRE(v == 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        REQUIRE(concurrent_count.load() == 1);
        auto v2 = --concurrent_count;
        REQUIRE(v2 == 0);
      } catch (...) {
        excpt = std::current_exception();
      }
    }};
    {
      --wait_count;
      while (0 != wait_count) {
      }
      {
        std::lock_guard<edm::SpinLock> g{lock};
        auto v = ++concurrent_count;
        REQUIRE(v == 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        REQUIRE(concurrent_count.load() == 1);
        auto v2 = --concurrent_count;
        REQUIRE(v2 == 0);
      }
    }
    t.join();
    if (excpt) {
      std::rethrow_exception(excpt);
    }
  }
}
