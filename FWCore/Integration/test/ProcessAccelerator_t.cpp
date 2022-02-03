#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "FWCore/TestProcessor/interface/TestProcessor.h"

#include <fmt/format.h>

#include <iostream>
#include <string_view>

static constexpr auto s_tag = "[ProcessAccelerator]";

namespace {
  std::string makeConfig(bool test2Enabled,
                         std::string_view test1,
                         std::string_view test2,
                         std::string_view accelerator) {
    const std::string appendTest2 = test2Enabled ? "self._enabled.append('test2')" : "";
    return fmt::format(
        R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

class ProcessAcceleratorTest(cms.ProcessAccelerator):
    def __init__(self):
        super(ProcessAcceleratorTest,self).__init__()
        self._labels = ["test1", "test2"]
        self._enabled = ["test1"]
        {}
    def labels(self):
        return self._labels
    def enabledLabels(self):
        return self._enabled

class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                cpu = cms.SwitchProducer.getCpu(),
                test1 = lambda accelerators: ("test1" in accelerators, 2),
                test2 = lambda accelerators: ("test2" in accelerators, 3),
            ), **kargs)

process = TestProcess()
process.options.accelerators = ["{}"]
process.ProcessAcceleratorTest = ProcessAcceleratorTest()
process.s = SwitchProducerTest(
   cpu = cms.EDProducer('IntProducer', ivalue = cms.int32(0)),
   test1 = {},
   test2 = {}
)
process.moduleToTest(process.s)
)_",
        appendTest2,
        accelerator,
        test1,
        test2);
  }
}  // namespace

TEST_CASE("Configuration", s_tag) {
  const std::string test1{"cms.EDProducer('IntProducer', ivalue = cms.int32(1))"};
  const std::string test2{"cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet())"};

  const std::string baseConfig_auto = makeConfig(true, test1, test2, "*");
  const std::string baseConfig_test1 = makeConfig(true, test1, test2, "test1");
  const std::string baseConfig_test2 = makeConfig(true, test1, test2, "test2");
  const std::string baseConfigTest2Disabled_auto = makeConfig(false, test1, test2, "*");
  const std::string baseConfigTest2Disabled_test1 = makeConfig(false, test1, test2, "test1");
  const std::string baseConfigTest2Disabled_test2 = makeConfig(false, test1, test2, "test2");

  SECTION("Configuration hash is not changed") {
    auto pset_auto = edm::readConfig(baseConfig_auto);
    auto pset_test1 = edm::readConfig(baseConfig_test1);
    auto pset_test2 = edm::readConfig(baseConfig_test2);
    auto psetTest2Disabled_auto = edm::readConfig(baseConfigTest2Disabled_auto);
    auto psetTest2Disabled_test1 = edm::readConfig(baseConfigTest2Disabled_test1);
    auto psetTest2Disabled_test2 = edm::readConfig(baseConfigTest2Disabled_test2);
    pset_auto->registerIt();
    pset_test1->registerIt();
    pset_test2->registerIt();
    psetTest2Disabled_auto->registerIt();
    psetTest2Disabled_test1->registerIt();
    psetTest2Disabled_test2->registerIt();
    REQUIRE(pset_auto->id() == pset_test1->id());
    REQUIRE(pset_auto->id() == pset_test2->id());
    REQUIRE(pset_auto->id() == psetTest2Disabled_auto->id());
    REQUIRE(pset_auto->id() == psetTest2Disabled_test1->id());
    REQUIRE(pset_auto->id() == psetTest2Disabled_test2->id());
  }

  edm::test::TestProcessor::Config config_auto{baseConfig_auto};
  edm::test::TestProcessor::Config config_test1{baseConfig_test1};
  edm::test::TestProcessor::Config config_test2{baseConfig_test2};
  edm::test::TestProcessor::Config configTest2Disabled_auto{baseConfigTest2Disabled_auto};
  edm::test::TestProcessor::Config configTest2Disabled_test1{baseConfigTest2Disabled_test1};
  edm::test::TestProcessor::Config configTest2Disabled_test2{baseConfigTest2Disabled_test2};

  SECTION("Base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config_auto)); }

  SECTION("No event data") {
    edm::test::TestProcessor tester(config_auto);
    REQUIRE_NOTHROW(tester.test());
  }

  SECTION("beginJob and endJob only") {
    edm::test::TestProcessor tester(config_auto);
    REQUIRE_NOTHROW(tester.testBeginAndEndJobOnly());
  }

  SECTION("Run with no LuminosityBlocks") {
    edm::test::TestProcessor tester(config_auto);
    REQUIRE_NOTHROW(tester.testRunWithNoLuminosityBlocks());
  }

  SECTION("LuminosityBlock with no Events") {
    edm::test::TestProcessor tester(config_auto);
    REQUIRE_NOTHROW(tester.testLuminosityBlockWithNoEvents());
  }

  SECTION("Test2 enabled, acclerators=*") {
    edm::test::TestProcessor tester(config_auto);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 2);
  }

  SECTION("Test2 enabled, acclerators=test1") {
    edm::test::TestProcessor tester(config_test1);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 1);
  }

  SECTION("Test2 enabled, acclerators=test2") {
    edm::test::TestProcessor tester(config_test2);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 2);
  }

  SECTION("Test2 disabled, accelerators=*") {
    edm::test::TestProcessor tester(configTest2Disabled_auto);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 1);
  }

  SECTION("Test2 disabled, accelerators=test1") {
    edm::test::TestProcessor tester(configTest2Disabled_test1);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 1);
  }

  SECTION("Test2 disabled, accelerators=test2") {
    REQUIRE_THROWS_WITH(
        edm::test::TestProcessor(configTest2Disabled_test2),
        Catch::Contains("The system has no compute accelerators that match the patterns") && Catch::Contains("test1"));
  }
}
