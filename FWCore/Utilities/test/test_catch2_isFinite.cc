#include "catch.hpp"

#include "FWCore/Utilities/interface/isFinite.h"

#include <limits>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>

TEST_CASE("test isFinite", "[isFinite]") {
  using namespace edm;
  typedef long double LD;

  //want to get a non-zero positive number that the compiler can't inline
  int n = getpid();
  double zero = atof("0");

  REQUIRE(isFinite(double(0)));
  REQUIRE(isFinite(float(0)));
  REQUIRE(isFinite(double(-3.14)));
  REQUIRE(isFinite(float(-3.14)));
  REQUIRE(!isFinite(std::sqrt(-double(n))));
  REQUIRE(!isFinite(std::sqrt(-float(n))));
  REQUIRE(!isFinite(1. / zero));
  REQUIRE(!isFinite(float(1.) / float(zero)));
  REQUIRE(!isFinite(-1. / zero));
  REQUIRE(!isFinite(-1.f / float(zero)));

  //

  REQUIRE(!isNotFinite(double(0)));
  REQUIRE(!isNotFinite(float(0)));
  REQUIRE(!isNotFinite(double(-3.14)));
  REQUIRE(!isNotFinite(float(-3.14)));
  REQUIRE(isNotFinite(std::sqrt(-double(n))));
  REQUIRE(isNotFinite(std::sqrt(-float(n))));
  REQUIRE(isNotFinite(-1.f / float(zero)));
  REQUIRE(isNotFinite(float(1.) / float(zero)));
  REQUIRE(isNotFinite(-1. / zero));
  REQUIRE(isNotFinite(-1.f / float(zero)));

  REQUIRE(!isNotFinite(LD(3.14)));
  REQUIRE(isNotFinite(-1 / LD(zero)));
  REQUIRE(isNotFinite(std::sqrt(-LD(n))));
}
