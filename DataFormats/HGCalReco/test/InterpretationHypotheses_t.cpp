#include <limits>
#include <vector>

#include <catch2/catch_all.hpp>

#include "DataFormats/HGCalReco/interface/InterpretationHypotheses.h"

TEST_CASE("InterpretationHypotheses keeps the hypotheses, their tracks and the conflicts",
          "[InterpretationHypotheses]") {
  using T = ticl::InterpretationHypotheses::Type;
  using V = ticl::InterpretationHypotheses::Vector;
  ticl::InterpretationHypotheses h;
  h.setTrackBonus(20.f);
  h.setSolverLimits(64, 1000000);
  const std::vector<int> track3{3}, jet{3, 5}, none{};
  const float inf = std::numeric_limits<float>::infinity();
  REQUIRE(h.add({T::Electron, track3, 1.5f, true, 10.f, 9.f, V(1.f, 2.f, 320.f)}) == 0);
  REQUIRE(h.add({T::Jet, jet, -inf, false, 30.f, 5.f, V(0.f, 1.f, 330.f)}) == 1);
  REQUIRE(h.add({T::Photon, none, 0.5f, false, 2.f, 2.f, V(1.f, 1.f, 321.f)}) == 2);
  h.addConflict(0, 1);
  h.addConflict(0, 2);

  REQUIRE(h.size() == 3);
  REQUIRE(h.type(1) == T::Jet);
  REQUIRE(h.tracks(0).size() == 1);
  REQUIRE(h.tracks(0)[0] == 3);
  REQUIRE(h.tracks(1).size() == 2);
  REQUIRE(h.tracks(1)[1] == 5);
  REQUIRE(h.tracks(2).empty());
  REQUIRE(h.solverWeight(0) == 21.5f);  // one track
  REQUIRE(h.solverWeight(1) == -inf);   // vetoed
  REQUIRE(h.solverWeight(2) == 0.5f);   // no track
  REQUIRE(h.accepted(0));
  REQUIRE_FALSE(h.accepted(2));
  REQUIRE(h.rawEmEnergy(0) == 9.f);
  REQUIRE(h.barycenter(2).z() == 321.f);
  REQUIRE(h.maxExactComponent() == 64);
  REQUIRE(h.searchBudget() == 1000000);
  REQUIRE(h.nConflicts() == 2);
  REQUIRE(h.conflict(1) == std::pair<unsigned int, unsigned int>(0, 2));

  // A hypothesis can take its tracks from another hypothesis.
  for (int k = 0; k < 100; ++k)
    h.add({T::ChargedHadron, h.tracks(1), 0.f, false, 1.f, 0.f, V(0.f, 0.f, 330.f)});
  REQUIRE(h.size() == 103);
  REQUIRE(h.tracks(102).size() == 2);
  REQUIRE(h.tracks(102)[0] == 3);
  REQUIRE(h.tracks(102)[1] == 5);
}
