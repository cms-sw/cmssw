#include "catch.hpp"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"


static constexpr auto s_tag = "[PFDisplacedVertexProducer]";

TEST_CASE("Standard checks of PFDisplacedVertexProducer", s_tag) {
  const std::string baseConfig{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
from RecoParticleFlow.PFTracking.particleFlowDisplacedVertex_cfi import particleFlowDisplacedVertex
process.toTest = particleFlowDisplacedVertex
process.moduleToTest(process.toTest)
)_"
  };

  const std::string fullConfig{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.load("MagneticField.Engine.uniformMagneticField_cfi")
process.load("Configuration.Geometry.GeometryExtended2018Reco_cff")
process.add_(cms.ESProducer("TrackerParametersESModule"))
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")


from RecoParticleFlow.PFTracking.particleFlowDisplacedVertex_cfi import particleFlowDisplacedVertex
process.toTest = particleFlowDisplacedVertex
process.moduleToTest(process.toTest)
)_"
  };
  
  edm::test::TestProcessor::Config config{ baseConfig };  
  SECTION("base configuration is OK") {
    REQUIRE_NOTHROW(edm::test::TestProcessor(config));
  }
  
  SECTION("No event data") {
    edm::test::TestProcessor::Config config{ fullConfig };  
    edm::test::TestProcessor tester(config);
    
    //The module ignores missing data products
    REQUIRE(tester.test().get<reco::PFDisplacedVertexCollection>()->empty());
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
