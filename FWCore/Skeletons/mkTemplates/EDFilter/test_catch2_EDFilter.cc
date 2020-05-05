#include "catch.hpp"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

static constexpr auto s_tag = "[__class__]";

TEST_CASE("Standard checks of __class__", s_tag) {
  const std::string baseConfig{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDFilter("__class__"
#necessary configuration parameters
 )
process.moduleToTest(process.toTest)
)_"
  };
  
  edm::test::TestProcessor::Config config{ baseConfig };  
  SECTION("base configuration is OK") {
    REQUIRE_NOTHROW(edm::test::TestProcessor(config));
  }
  
  SECTION("No event data") {
    edm::test::TestProcessor tester(config);
    
    REQUIRE_THROWS_AS(tester.test(), cms::Exception);
    //If the module does not throw when given no data, substitute 
    //REQUIRE_NOTHROW for REQUIRE_THROWS_AS
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

}

//Add additional TEST_CASEs to exercise the modules capabilities
