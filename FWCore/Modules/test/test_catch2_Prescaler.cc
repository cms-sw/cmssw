#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "catch.hpp"

static constexpr auto s_tag = "[Prescaler]";
TEST_CASE("Check parameters of Prescaler", s_tag) {
  const std::string baseConfig{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.prescale = cms.EDFilter('Prescaler',
                                 prescaleFactor =cms.int32(3),
                                 prescaleOffset = cms.int32(offset))
process.moduleToTest(process.prescale)
)_"
  };
  
  SECTION("prescaleFactor sets how many events to fail before success") {
    edm::test::TestProcessor::Config config{
      "offset=1\n"+baseConfig 
    };
    
    edm::test::TestProcessor tester(config);
    REQUIRE(tester.test().modulePassed());
    REQUIRE_FALSE(tester.test().modulePassed());
    REQUIRE_FALSE(tester.test().modulePassed());
    REQUIRE(tester.test().modulePassed());
    REQUIRE_FALSE(tester.test().modulePassed());
    REQUIRE_FALSE(tester.test().modulePassed());
    REQUIRE(tester.test().modulePassed());
  }
  
  SECTION("prescaleOffset sets which event succeeds") {
    edm::test::TestProcessor::Config config{
      "offset=2\n"+baseConfig
    };

    edm::test::TestProcessor tester(config);
    REQUIRE_FALSE(tester.test().modulePassed());
    REQUIRE(tester.test().modulePassed());
    REQUIRE_FALSE(tester.test().modulePassed());
    REQUIRE_FALSE(tester.test().modulePassed());
    REQUIRE(tester.test().modulePassed());    
  }
}

TEST_CASE("No default parameters of Prescaler",s_tag) {
  const std::string config{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.prescale = cms.EDFilter('Prescaler')
process.moduleToTest(process.prescale)
)_"
  };
  
  REQUIRE_THROWS_AS( edm::test::TestProcessor(config), cms::Exception);
}

