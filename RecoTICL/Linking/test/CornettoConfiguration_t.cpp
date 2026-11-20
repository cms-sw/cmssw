#include <string>

#include <catch2/catch_all.hpp>

#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace {
  // A TracksterLinksProducer with the Cornetto plugin at its description defaults, then the overrides.
  edm::test::TestProcessor::Config cornettoConfig(const std::string& overrides) {
    return edm::test::TestProcessor::Config{std::string{R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
from RecoTICL.Linking.tracksterLinksProducer_cfi import tracksterLinksProducer
process.linker = tracksterLinksProducer.clone(linkingPSet = cms.PSet(type = cms.string('Cornetto')))
)_"} + overrides + "\nprocess.moduleToTest(process.linker)\n"};
  }

  void requireInvalid(const std::string& overrides) {
    INFO(overrides);
    REQUIRE_THROWS_WITH(edm::test::TestProcessor(cornettoConfig(overrides)),
                        Catch::Matchers::ContainsSubstring("TracksterLinkingbyCornetto: invalid parameters"));
  }

  void requireValid(const std::string& overrides) {
    INFO(overrides);
    REQUIRE_NOTHROW(edm::test::TestProcessor(cornettoConfig(overrides)));
  }

  const std::string kPSet = "process.linker.linkingPSet.";
}  // namespace

TEST_CASE("The Cornetto plugin builds with its description defaults", "[CornettoConfiguration]") {
  requireValid("");
  requireValid(kPSet + "etaWindow = cms.float(3.09)");
  requireValid(kPSet + "typeVetoProbability = cms.float(0)");
  requireValid(kPSet + "typeVetoProbability = cms.float(0.5)");
  requireValid(kPSet + "typeVetoProbability = cms.float(1)");
  requireValid(kPSet + "maxBackwardDistance = cms.float(0)");
  requireValid(kPSet + "timeCompatibilityNSigma = cms.float(-1)");
}

TEST_CASE("The Cornetto plugin refuses invalid parameters", "[CornettoConfiguration]") {
  SECTION("etaWindow must be in (0, pi - 2 pi / 128)") {
    requireInvalid(kPSet + "etaWindow = cms.float(0)");
    requireInvalid(kPSet + "etaWindow = cms.float(3.1)");
  }
  SECTION("Cone") {
    requireInvalid(kPSet + "maxLongitudinalDistance = cms.float(0)");
    requireInvalid(kPSet + "maxBackwardDistance = cms.float(-1)");
    requireInvalid(kPSet + "transverseRadius0 = cms.float(-1)");
    requireInvalid(kPSet + "transverseSlope = cms.float(-0.01)");
    requireInvalid(kPSet + "emTransverseRadius0 = cms.float(5)");
    requireInvalid(kPSet + "emTransverseSlope = cms.float(0.06)");
  }
  SECTION("Merge cone") {
    requireInvalid(kPSet + "mergeMaxLongitudinalDistance = cms.float(0)");
    requireInvalid(kPSet + "mergeTransverseRadius0 = cms.float(-1)");
    requireInvalid(kPSet + "mergeEMTransverseRadius0 = cms.float(5)");
  }
  SECTION("Time resolution floor") { requireInvalid(kPSet + "timeResolutionFloor = cms.float(-0.01)"); }
  SECTION("typeVetoProbability must be 0 or in [0.5, 1]") {
    requireInvalid(kPSet + "typeVetoProbability = cms.float(0.4)");
    requireInvalid(kPSet + "typeVetoProbability = cms.float(1.1)");
  }
  SECTION("emCategories: at least one value, each in [0, 7], no duplicates") {
    requireInvalid(kPSet + "emCategories = cms.vint32()");
    requireInvalid(kPSet + "emCategories = cms.vint32(8)");
    requireInvalid(kPSet + "emCategories = cms.vint32(-1)");
    requireInvalid(kPSet + "emCategories = cms.vint32(0, 0)");
  }
}
