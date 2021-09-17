#include "CondFormats/EcalObjects/interface/EcalPulseSymmCovariances.h"

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include <iostream>

TEST_CASE("EcalPulseSymmCovariance testing", "[EcalPulseSymmCovariance]") {
  EcalPulseSymmCovariance covMutable;
  //fill with accending values
  float const vMin = 0.5f;
  float v = vMin;

  std::set<float> values;
  for (auto& entry : covMutable.covval) {
    entry = v;
    values.insert(v);
    v += 1.f;
  }

  float const vMax = v - 1.f;

  auto const& cov = covMutable;

  SECTION("Check symmetry") {
    for (int i = 0; i < EcalPulseShape::TEMPLATESAMPLES; ++i) {
      for (int j = 0; j < EcalPulseShape::TEMPLATESAMPLES; ++j) {
        REQUIRE(cov.val(i, j) == cov.val(j, i));
      }
    }
  }
  SECTION("Check index coverage") {
    std::vector<bool> hitIndices(std::size(cov.covval), false);
    for (int i = 0; i < EcalPulseShape::TEMPLATESAMPLES; ++i) {
      for (int j = 0; j < EcalPulseShape::TEMPLATESAMPLES; ++j) {
        hitIndices[cov.indexFor(i, j)] = true;
      }
    }
    for (auto indx : hitIndices) {
      REQUIRE(indx == true);
    }
  }
  SECTION("Check bounds") {
    for (int i = 0; i < EcalPulseShape::TEMPLATESAMPLES; ++i) {
      for (int j = 0; j < EcalPulseShape::TEMPLATESAMPLES; ++j) {
        REQUIRE((unsigned)cov.indexFor(i, j) < std::size(cov.covval));
      }
    }
  }

  SECTION("Check known values") {
    for (int i = 0; i < EcalPulseShape::TEMPLATESAMPLES; ++i) {
      for (int j = 0; j < EcalPulseShape::TEMPLATESAMPLES; ++j) {
        REQUIRE(vMin <= cov.val(i, j));
        REQUIRE(cov.val(i, j) <= vMax);
        REQUIRE(values.end() != values.find(cov.val(i, j)));
      }
    }
  }

  SECTION("Check filling") {
    float v = vMin;
    EcalPulseSymmCovariance covNew;
    for (int i = 0; i < EcalPulseShape::TEMPLATESAMPLES; ++i) {
      for (int j = 0; j < EcalPulseShape::TEMPLATESAMPLES; ++j) {
        covNew.val(i, j) = v;
        REQUIRE(v == covNew.val(i, j));
      }
    }
  }
}
