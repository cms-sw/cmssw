#include "catch.hpp"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"

#include <fmt/format.h>

#include <iostream>
#include <string_view>

static constexpr auto s_tag = "[ProcessAcceleratorCUDA]";

namespace {
  std::string makeConfig(std::string_view cpu, std::string_view cuda, std::string_view accelerator) {
    return fmt::format(
        R"_(import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.options.accelerators = [{}]

process.load("HeterogeneousCore.CUDACore.ProcessAcceleratorCUDA_cfi")
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import *

process.s = SwitchProducerCUDA(
   cpu = {},
   cuda = {}
)
process.p = cms.Path(process.s)
)_",
        accelerator,
        cpu,
        cuda);
  }
}  // namespace

TEST_CASE("Configuration", s_tag) {
  const std::string test1{"cms.EDProducer('IntProducer', ivalue = cms.int32(1))"};
  const std::string test2{"cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet())"};

  const std::string baseConfig_auto = makeConfig(test1, test2, "'*'");
  const std::string baseConfig_cpu = makeConfig(test1, test2, "'cpu'");
  const std::string baseConfig_cuda = makeConfig(test1, test2, "'gpu-nvidia'");

  SECTION("Configuration hash is not changed") {
    std::unique_ptr<edm::ParameterSet> pset_auto, pset_cpu, pset_cuda;
    edm::makeParameterSets(baseConfig_auto, pset_auto);
    edm::makeParameterSets(baseConfig_cpu, pset_cpu);
    edm::makeParameterSets(baseConfig_cuda, pset_cuda);
    pset_auto->registerIt();
    pset_cpu->registerIt();
    pset_cuda->registerIt();
    REQUIRE(pset_auto->id() == pset_cpu->id());
    REQUIRE(pset_auto->id() == pset_cuda->id());
  }
}
