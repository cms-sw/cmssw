#include "FWCore/Utilities/interface/HRRealTime.h"

#include <ctime>
#include <cmath>
#include <typeinfo>
#include <iostream>

#include <catch2/catch_all.hpp>

namespace {

  double gcrap = 0;
  void waiste() {
    for (double i = 1; i < 100000; i++)
      gcrap += std::log(std::sqrt(i));
  }
}  // namespace

// Helper for timing checks

// Replace CppUnit assertions with Catch2 REQUIRE

template <typename S>
void checkTime(S source, bool hr) {
  typedef edm::HRTimeDiffType T;

  T i = source();
  REQUIRE(!(i < 0));

  waiste();

  T a = source();
  T b = source();
  REQUIRE(!((a - i) < 0));
  REQUIRE(!((b - a) < 0));
  if (hr)
    REQUIRE(a > i);  // not obvious if low resolution

  waiste();

  T c = source();
  double d = double(source() - c);
  REQUIRE(!(d < 0));

  T e = source();
  REQUIRE(!((c - i) < (b - i)));
  REQUIRE(!((e - i) < (c - i)));
  if (hr)
    REQUIRE((e - i) > (b - i));  // not obvious if low resolution...
}

TEST_CASE("HRTime", "[HRTime]") {
  SECTION("stdclock") {
    std::cout << "checking source std::clock" << std::endl;
    checkTime(&std::clock, false);
  }
  SECTION("RealTime") {
    std::cout << "checking source edm::hrRealTime" << std::endl;
    checkTime(&edm::hrRealTime, true);
  }
}
