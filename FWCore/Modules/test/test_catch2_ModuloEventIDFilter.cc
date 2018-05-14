#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "catch.hpp"

static constexpr auto s_tag = "[ModuloEventIDFilter]";

TEST_CASE("Check parameters of ModuloEventIDFilter", s_tag) {
  const std::string baseConfig{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDFilter('ModuloEventIDFilter',
                                 modulo =cms.uint32(3),
                                 offset = cms.uint32(offset))
process.moduleToTest(process.toTest)
)_"
  };
  
  SECTION("modulo sets how many events to fail before success") {
    edm::test::TestProcessor::Config config{
      "offset=1\n"+baseConfig 
    };
    
    edm::test::TestProcessor tester(config);
    tester.setEventNumber(1);
    REQUIRE(tester.test().modulePassed());
    REQUIRE_FALSE(tester.test().modulePassed());
    REQUIRE_FALSE(tester.test().modulePassed());
    REQUIRE(tester.test().modulePassed());
    REQUIRE_FALSE(tester.test().modulePassed());
    REQUIRE_FALSE(tester.test().modulePassed());
    REQUIRE(tester.test().modulePassed());
    tester.setEventNumber(10);
    REQUIRE(tester.test().modulePassed());
    tester.setEventNumber(13);
    REQUIRE(tester.test().modulePassed());
    tester.setEventNumber(15);
    REQUIRE_FALSE(tester.test().modulePassed());
  }
  
  SECTION("offset sets which event succeeds") {
    edm::test::TestProcessor::Config config{
      "offset=2\n"+baseConfig
    };

    edm::test::TestProcessor tester(config);
    tester.setEventNumber(1);
    REQUIRE_FALSE(tester.test().modulePassed());
    REQUIRE(tester.test().modulePassed());
    REQUIRE_FALSE(tester.test().modulePassed());
    REQUIRE_FALSE(tester.test().modulePassed());
    REQUIRE(tester.test().modulePassed());
    tester.setEventNumber(8);
    REQUIRE(tester.test().modulePassed());
    tester.setEventNumber(11);
    REQUIRE(tester.test().modulePassed());
    tester.setEventNumber(13);
    REQUIRE_FALSE(tester.test().modulePassed());
  }
}

TEST_CASE("No default parameters of ModuloEventIDFilter",s_tag) {
  const std::string config{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDFilter('ModuloEventIDFilter')
process.moduleToTest(process.toTest)
)_"
  };
  
  REQUIRE_THROWS_AS( edm::test::TestProcessor(config), cms::Exception);
}

