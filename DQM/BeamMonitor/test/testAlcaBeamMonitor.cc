#include "FWCore/TestProcessor/interface/TestProcessor.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("AlcaBeamMonitor tests", "[AlcaBeamMonitor]") {
  //The python configuration
  edm::test::TestProcessor::Config config{
      R"_(from FWCore.TestProcessor.TestProcess import *
from DQM.BeamMonitor.AlcaBeamMonitor_cfi import AlcaBeamMonitor
process = TestProcess()
process.beamMonitor = AlcaBeamMonitor
process.moduleToTest(process.beamMonitor)
process.add_(cms.Service("DQMStore"))
)_"};

  SECTION("Run with no Lumis") {
    edm::test::TestProcessor tester{config};
    tester.testRunWithNoLuminosityBlocks();
    //get here without an exception or crashing
    REQUIRE(true);
  };
}
