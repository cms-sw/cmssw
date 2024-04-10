#include "catch.hpp"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

static constexpr auto s_tag = "[NanoAODDQM]";

TEST_CASE("Standard checks of NanoAODDQM", s_tag) {
  const std::string baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("NanoAODDQM",
  vplots = cms.PSet( 
    TEST = cms.PSet(
       sels = cms.PSet( Good = cms.string("foo > 0 && abs(bar) > 1.3") ),
       plots = cms.VPSet(
          cms.PSet( name = cms.string("foo"), kind = cms.string("none"))
       )
    )
  )
 )
process.add_(cms.Service("DQMStore"))
process.moduleToTest(process.toTest)
)_"};

  edm::test::TestProcessor::Config config{baseConfig};
  SECTION("base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

  SECTION("No event data") {
    edm::test::TestProcessor tester(config);

    REQUIRE_NOTHROW(tester.test());
  }

  SECTION("With good event data") {
    auto token = config.produces<nanoaod::FlatTable>("test");
    edm::test::TestProcessor tester(config);
    auto table = std::make_unique<nanoaod::FlatTable>(1, "TEST", false);
    table->addColumn<float>("foo", std::vector<float>(1, 3.0f), "is foo");
    table->addColumn<float>("bar", std::vector<float>(1, 5.0f), "is bar");
    REQUIRE_NOTHROW(tester.test(std::make_pair(token, std::move(table))));
  }

  SECTION("With wrong column name") {
    auto token = config.produces<nanoaod::FlatTable>("test");
    edm::test::TestProcessor tester(config);
    auto table = std::make_unique<nanoaod::FlatTable>(1, "TEST", false);
    table->addColumn<float>("Foo", std::vector<float>(1, 3.0f), "is foo");
    table->addColumn<float>("bar", std::vector<float>(1, 5.0f), "is bar");
    REQUIRE_THROWS_AS(tester.test(std::make_pair(token, std::move(table))), cms::Exception);
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

  SECTION("Use Row Method") {
    const std::string baseConfig{
        R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer("NanoAODDQM",
  vplots = cms.PSet( 
    TEST = cms.PSet(
       sels = cms.PSet( Good = cms.string("foo > 0 && row == 0") ),
       plots = cms.VPSet(
          cms.PSet( name = cms.string("foo"), kind = cms.string("none"))
       )
    )
  )
 )
process.add_(cms.Service("DQMStore"))
process.moduleToTest(process.toTest)
)_"};

    edm::test::TestProcessor::Config config{baseConfig};

    auto token = config.produces<nanoaod::FlatTable>("test");
    edm::test::TestProcessor tester(config);
    auto table = std::make_unique<nanoaod::FlatTable>(1, "TEST", false);
    table->addColumn<float>("foo", std::vector<float>(1, 3.0f), "is foo");
    REQUIRE_NOTHROW(tester.test(std::make_pair(token, std::move(table))));
  }
}

//Add additional TEST_CASEs to exercise the modules capabilities
