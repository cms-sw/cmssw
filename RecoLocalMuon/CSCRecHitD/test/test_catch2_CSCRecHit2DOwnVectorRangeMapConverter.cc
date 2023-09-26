#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "catch.hpp"

static constexpr auto s_tag = "[CSCRecHit2DOwnVectorRangeMapConverter]";

TEST_CASE("Standard checks of CSCRecHit2DOwnVectorRangeMapConverter", s_tag) {
  const std::string baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("CSCRecHit2DOwnVectorRangeMapConverter",
get = cms.InputTag("src")
 )
process.moduleToTest(process.toTest)
)_"};

  edm::test::TestProcessor::Config config{baseConfig};
  using OldRange = edm::RangeMap<CSCDetId, edm::OwnVector<CSCRecHit2D>>;
  auto token = config.produces<OldRange>("src");

  SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

  SECTION("No event data") {
    edm::test::TestProcessor tester(config);

    REQUIRE_THROWS_AS(tester.test(), cms::Exception);
  }

  SECTION("beginJob and endJob only") {
    edm::test::TestProcessor tester(config);

    REQUIRE_NOTHROW(tester.testBeginAndEndJobOnly());
  }

  SECTION("Run with no LuminosityBlocks") {
    edm::test::TestProcessor tester(config);

    REQUIRE_NOTHROW(tester.testRunWithNoLuminosityBlocks());
  }

  SECTION("LuminosityBlock with no Events") {
    edm::test::TestProcessor tester(config);

    REQUIRE_NOTHROW(tester.testLuminosityBlockWithNoEvents());
  }
  SECTION("Transform old data") {
    edm::test::TestProcessor tester(config);

    auto oldRange = std::make_unique<OldRange>();
    std::vector<CSCRecHit2D> v(3);
    CSCDetId id1(1, 0, 0, 0);
    CSCDetId id2(1, 1, 0, 0);
    CSCDetId id3(2, 0, 0, 0);
    std::vector<CSCDetId> ordered;
    ordered.reserve(3);
    ordered.push_back(id1);
    ordered.push_back(id2);
    ordered.push_back(id3);
    std::sort(ordered.begin(), ordered.end(), std::less<CSCDetId>());

    oldRange->put(id1, v.begin(), v.begin() + 1);
    oldRange->put(id2, v.begin(), v.begin() + 2);
    oldRange->put(id3, v.begin(), v.end());
    oldRange->post_insert();
    REQUIRE(oldRange->size() == 6);
    REQUIRE(oldRange->id_size() == 3);
    REQUIRE(oldRange->ids() == ordered);

    auto event = tester.test(std::make_pair(token, std::move(oldRange)));

    auto newRange = event.get<CSCRecHit2DCollection>();
    REQUIRE(newRange->size() == 6);
    REQUIRE(newRange->id_size() == 3);
    REQUIRE(newRange->ids() == ordered);
  }
}
