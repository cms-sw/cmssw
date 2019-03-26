#include "catch.hpp"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDATest/interface/CUDAThing.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"

#include <iostream>

static constexpr auto s_tag = "[TestCUDAProducerGPUFirst]";

TEST_CASE("Standard checks of TestCUDAProducerGPUFirst", s_tag) {
  const std::string baseConfig{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")
process.toTest = cms.EDProducer("TestCUDAProducerGPUFirst")
process.moduleToTest(process.toTest)
)_"
  };

  edm::test::TestProcessor::Config config{ baseConfig };
  SECTION("base configuration is OK") {
    REQUIRE_NOTHROW(edm::test::TestProcessor(config));
  }

  SECTION("No event data") {
    edm::test::TestProcessor tester(config);

    REQUIRE_NOTHROW(tester.test());
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

TEST_CASE("TestCUDAProducerGPUFirst operation", s_tag) {
  const std::string baseConfig{
R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")
process.toTest = cms.EDProducer("TestCUDAProducerGPUFirst")
process.moduleToTest(process.toTest)
)_"
  };
  edm::test::TestProcessor::Config config{ baseConfig };

  exitSansCUDADevices();

  constexpr int defaultDevice = 0;

  SECTION("Produce") {
    edm::test::TestProcessor tester{config};
    auto event = tester.test();
    auto prod = event.get<CUDAProduct<CUDAThing> >();
    REQUIRE(prod->device() == defaultDevice);
    auto ctx = CUDAScopedContext(*prod);
    const CUDAThing& thing = ctx.get(*prod);
    const float *data = thing.get();
    REQUIRE(data != nullptr);

    float firstElements[10];
    cuda::memory::async::copy(firstElements, data, sizeof(float)*10, prod->stream().id());

    std::cout << "Synchronizing with CUDA stream" << std::endl;
    auto stream = prod->stream();
    stream.synchronize();
    std::cout << "Synchronized" << std::endl;
    REQUIRE(firstElements[0] == 0.f);
    REQUIRE(firstElements[1] == 1.f);
    REQUIRE(firstElements[9] == 9.f);
  }
};
