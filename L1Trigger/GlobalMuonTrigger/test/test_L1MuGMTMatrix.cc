#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatrix.h"

namespace {
  template <int I, int J, typename T>
  bool compare(const L1MuGMTMatrix<T>& iMatrix, std::array<T, I * J> const& iValues) {
    for (int i = 0; i < I; ++i) {
      for (int j = 0; j < J; ++j) {
        if (iMatrix(i, j) != iValues[j + i * J]) {
          return false;
        }
      }
    }
    return true;
  }
}  // namespace

TEST_CASE("Test L1MuGMTMatrix", "L1MuGMTMatrix") {
  SECTION("empty") {
    L1MuGMTMatrix<int> m(2, 3, 0);
    REQUIRE(compare<2, 3>(m, {{0, 0, 0, 0, 0, 0}}) == true);
  }

  SECTION("set all to 1") {
    L1MuGMTMatrix<int> m(2, 3, 1);
    REQUIRE(compare<2, 3>(m, {{1, 1, 1, 1, 1, 1}}) == true);
  }

  SECTION("set different values") {
    L1MuGMTMatrix<int> m(2, 3, 0);
    m.set(0, 0, 0);
    m.set(0, 1, 1);
    m.set(0, 2, 2);

    m.set(1, 0, 3);
    m.set(1, 1, 4);
    m.set(1, 2, 5);

    REQUIRE(compare<2, 3>(m, {{0, 1, 2, 3, 4, 5}}) == true);

    REQUIRE(m.isMax(0, 0) == false);
    REQUIRE(m.isMax(0, 1) == false);
    REQUIRE(m.isMax(0, 2) == false);
    REQUIRE(m.isMax(1, 0) == false);
    REQUIRE(m.isMax(1, 1) == false);
    REQUIRE(m.isMax(1, 2) == true);

    REQUIRE(m.isMin(0, 0) == true);
    REQUIRE(m.isMin(0, 1) == false);
    REQUIRE(m.isMin(0, 2) == false);

    REQUIRE(m.isMin(1, 0) == false);
    REQUIRE(m.isMin(1, 1) == false);
    REQUIRE(m.isMin(1, 2) == false);
  }

  SECTION("Copy and operator=") {
    L1MuGMTMatrix<int> m(2, 3, 0);
    m.set(0, 0, 0);
    m.set(0, 1, 1);
    m.set(0, 2, 2);

    m.set(1, 0, 3);
    m.set(1, 1, 4);
    m.set(1, 2, 5);

    L1MuGMTMatrix<int> cp(m);

    REQUIRE(compare<2, 3>(cp, {{0, 1, 2, 3, 4, 5}}) == true);

    L1MuGMTMatrix<int> opEq(2, 3, 0);
    opEq = m;

    REQUIRE(compare<2, 3>(opEq, {{0, 1, 2, 3, 4, 5}}) == true);
  }

  SECTION("Arithmetics") {
    L1MuGMTMatrix<int> m(2, 3, 0);
    m.set(0, 0, 0);
    m.set(0, 1, 1);
    m.set(0, 2, 2);

    m.set(1, 0, 3);
    m.set(1, 1, 4);
    m.set(1, 2, 5);

    SECTION("Scalar addition") { REQUIRE(compare<2, 3>(m += 1, {{1, 2, 3, 4, 5, 6}}) == true); }

    SECTION("Scalar multiplication") { REQUIRE(compare<2, 3>(m *= 2, {{0, 2, 4, 6, 8, 10}}) == true); }

    SECTION("Matrix addition") {
      L1MuGMTMatrix<int> cp(m);
      REQUIRE(compare<2, 3>(m += cp, {{0, 2, 4, 6, 8, 10}}) == true);
    }
  }

  SECTION("Look for non 0") {
    L1MuGMTMatrix<int> m(2, 3, 0);

    m.set(0, 0, 0);
    m.set(0, 1, 0);
    m.set(0, 2, 2);

    m.set(1, 0, 0);
    m.set(1, 1, 0);
    m.set(1, 2, 0);

    REQUIRE(m.colAny(0) == -1);
    REQUIRE(m.colAny(1) == -1);
    REQUIRE(m.colAny(2) == 0);

    REQUIRE(m.rowAny(0) == 2);
    REQUIRE(m.rowAny(1) == -1);
  }
}
