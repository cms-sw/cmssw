/*----------------------------------------------------------------------
Test program for edm::TypeID class.
 ----------------------------------------------------------------------*/

#include <cassert>
#include <catch2/catch_all.hpp>
#include "FWCore/Utilities/interface/CallOnceNoWait.h"
#include "FWCore/Utilities/interface/CallNTimesNoWait.h"
#include <thread>
#include <atomic>
#include <vector>

TEST_CASE("CallXNoWait", "[CallXNoWait]") {
  SECTION("onceTest") {
    edm::CallOnceNoWait guard;
    unsigned int iCount = 0;
    guard([&iCount]() { ++iCount; });
    REQUIRE(iCount == 1u);
    guard([&iCount]() { ++iCount; });
    REQUIRE(iCount == 1u);
  }

  SECTION("nTimesTest") {
    edm::CallNTimesNoWait guard{3};
    unsigned int iCount = 0;
    for (unsigned int i = 0; i < 6; ++i) {
      guard([&iCount]() { ++iCount; });
      if (i < 3) {
        REQUIRE(iCount == i + 1);
      } else {
        REQUIRE(iCount == 3u);
      }
    }
  }

  SECTION("onceThreadedTest") {
    edm::CallOnceNoWait guard;
    std::atomic<unsigned int> iCount{0};
    std::vector<std::thread> threads;
    std::atomic<bool> start{false};
    for (unsigned int i = 0; i < 4; ++i) {
      threads.emplace_back([&guard, &iCount, &start]() {
        while (!start) {
        }
        guard([&iCount]() { ++iCount; });
      });
    }
    REQUIRE(iCount == 0u);
    start = true;
    for (auto& t : threads) {
      t.join();
    }
    REQUIRE(iCount == 1u);
  }

  SECTION("nTimesThreadedTest") {
    const unsigned short kMaxTimes = 3;
    edm::CallNTimesNoWait guard(kMaxTimes);
    std::atomic<unsigned int> iCount{0};
    std::vector<std::thread> threads;
    std::atomic<bool> start{false};
    for (unsigned int i = 0; i < 2 * kMaxTimes; ++i) {
      threads.emplace_back([&guard, &iCount, &start]() {
        while (!start) {
        }
        guard([&iCount]() { ++iCount; });
      });
    }
    REQUIRE(iCount == 0u);
    start = true;
    for (auto& t : threads) {
      t.join();
    }
    REQUIRE(iCount == kMaxTimes);
  }
}
