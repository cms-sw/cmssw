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
  std::string makeSwitchConfig(bool test2Enabled,
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

  std::string makeResolverConfig(bool enableOther, std::string_view accelerator, std::string_view variant) {
    const std::string appendOther = enableOther ? "self._enabled.append('other')" : "";
    const std::string explicitVariant =
        variant.empty() ? std::string(variant) : fmt::format(", variant=cms.untracked.string('{}')", variant);
    return fmt::format(
        R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

class ModuleTypeResolverTest:
    def __init__(self, accelerators):
        self._variants = []
        if "other" in accelerators:
            self._variants.append("other")
        if "cpu" in accelerators:
            self._variants.append("cpu")
        if len(self._variants) == 0:
            raise cms.EDMException(cms.edm.errors.UnavailableAccelerator, "No 'cpu' or 'other' accelerator available")

    def plugin(self):
        return "edm::test::ConfigurableTestTypeResolverMaker"

    def setModuleVariant(self, module):
        if "generic::" in module.type_():
            if hasattr(module, "variant"):
                if module.variant.value() not in self._variants:
                    raise cms.EDMException(cms.edm.errors.UnavailableAccelerator, "Module {{}} has the Test variant set explicitly to {{}}, but its accelerator is not available for the job".format(module.label_(), module.variant.value()))
            else:
                module.variant = cms.untracked.string(self._variants[0])

class ProcessAcceleratorTest(cms.ProcessAccelerator):
    def __init__(self):
        super(ProcessAcceleratorTest,self).__init__()
        self._labels = ["other"]
        self._enabled = []
        {}
    def labels(self):
        return self._labels
    def enabledLabels(self):
        return self._enabled
    def moduleTypeResolver(self, accelerators):
        return ModuleTypeResolverTest(accelerators)

process = TestProcess()
process.options.accelerators = ["{}"]
process.ProcessAcceleratorTest = ProcessAcceleratorTest()
process.m = cms.EDProducer("generic::IntProducer",
    valueCpu = cms.int32(1), valueOther = cms.int32(2)
    {}
)
process.moduleToTest(process.m)
)_",
        appendOther,
        accelerator,
        explicitVariant);
  }
}  // namespace

TEST_CASE("Configuration with SwitchProducer", s_tag) {
  const std::string test1{"cms.EDProducer('IntProducer', ivalue = cms.int32(1))"};
  const std::string test2{"cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet())"};

  const std::string baseConfig_auto = makeSwitchConfig(true, test1, test2, "*");
  const std::string baseConfig_test1 = makeSwitchConfig(true, test1, test2, "test1");
  const std::string baseConfig_test2 = makeSwitchConfig(true, test1, test2, "test2");
  const std::string baseConfigTest2Disabled_auto = makeSwitchConfig(false, test1, test2, "*");
  const std::string baseConfigTest2Disabled_test1 = makeSwitchConfig(false, test1, test2, "test1");
  const std::string baseConfigTest2Disabled_test2 = makeSwitchConfig(false, test1, test2, "test2");

  SECTION("Configuration hash is not changed") {
    std::unique_ptr<edm::ParameterSet> pset_auto, pset_test1, pset_test2;
    std::unique_ptr<edm::ParameterSet> psetTest2Disabled_auto, psetTest2Disabled_test1, psetTest2Disabled_test2;
    edm::makeParameterSets(baseConfig_auto, pset_auto);
    edm::makeParameterSets(baseConfig_test1, pset_test1);
    edm::makeParameterSets(baseConfig_test2, pset_test2);
    edm::makeParameterSets(baseConfigTest2Disabled_auto, psetTest2Disabled_auto);
    edm::makeParameterSets(baseConfigTest2Disabled_test1, psetTest2Disabled_test1);
    edm::makeParameterSets(baseConfigTest2Disabled_test2, psetTest2Disabled_test2);
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

TEST_CASE("Configuration with ModuleTypeResolver", s_tag) {
  const std::string baseConfig_auto = makeResolverConfig(true, "*", "");
  const std::string baseConfig_cpu = makeResolverConfig(true, "cpu", "");
  const std::string baseConfig_other = makeResolverConfig(true, "other", "");
  const std::string baseConfig_cpuExplicit = makeResolverConfig(true, "*", "cpu");
  const std::string baseConfig_otherExplicit = makeResolverConfig(true, "*", "other");
  const std::string baseConfigOtherDisabled_auto = makeResolverConfig(false, "*", "");
  const std::string baseConfigOtherDisabled_cpu = makeResolverConfig(false, "cpu", "");
  const std::string baseConfigOtherDisabled_other = makeResolverConfig(false, "other", "");

  SECTION("Configuration hash is not changed") {
    std::unique_ptr<edm::ParameterSet> pset_auto, pset_cpu, pset_other, pset_cpuExplicit;
    std::unique_ptr<edm::ParameterSet> pset_otherExplicit, psetOtherDisabled_auto, psetOtherDisabled_cpu,
        psetOtherDisabled_other;
    edm::makeParameterSets(baseConfig_auto, pset_auto);
    edm::makeParameterSets(baseConfig_cpu, pset_cpu);
    edm::makeParameterSets(baseConfig_other, pset_other);
    edm::makeParameterSets(baseConfig_cpuExplicit, pset_cpuExplicit);
    edm::makeParameterSets(baseConfig_otherExplicit, pset_otherExplicit);
    edm::makeParameterSets(baseConfigOtherDisabled_auto, psetOtherDisabled_auto);
    edm::makeParameterSets(baseConfigOtherDisabled_cpu, psetOtherDisabled_cpu);
    REQUIRE_THROWS_WITH(edm::makeParameterSets(baseConfigOtherDisabled_other, psetOtherDisabled_other),
                        Catch::Contains("UnavailableAccelerator"));
    pset_auto->registerIt();
    pset_cpu->registerIt();
    pset_other->registerIt();
    pset_cpuExplicit->registerIt();
    pset_otherExplicit->registerIt();
    psetOtherDisabled_auto->registerIt();
    psetOtherDisabled_cpu->registerIt();
    REQUIRE(pset_auto->id() == pset_cpu->id());
    REQUIRE(pset_auto->id() == pset_other->id());
    REQUIRE(pset_auto->id() == pset_cpuExplicit->id());
    REQUIRE(pset_auto->id() == pset_otherExplicit->id());
    REQUIRE(pset_auto->id() == psetOtherDisabled_auto->id());
    REQUIRE(pset_auto->id() == psetOtherDisabled_cpu->id());
  }

  edm::test::TestProcessor::Config config_auto{baseConfig_auto};
  edm::test::TestProcessor::Config config_cpu{baseConfig_cpu};
  edm::test::TestProcessor::Config config_other{baseConfig_other};
  edm::test::TestProcessor::Config config_cpuExplicit{baseConfig_cpuExplicit};
  edm::test::TestProcessor::Config config_otherExplicit{baseConfig_otherExplicit};
  edm::test::TestProcessor::Config configOtherDisabled_auto{baseConfigOtherDisabled_auto};
  edm::test::TestProcessor::Config configOtherDisabled_cpu{baseConfigOtherDisabled_cpu};

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

  SECTION("Other enabled, acclerators=*") {
    edm::test::TestProcessor tester(config_auto);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 2);
  }

  SECTION("Other enabled, acclerators=cpu") {
    edm::test::TestProcessor tester(config_cpu);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 1);
  }

  SECTION("Other enabled, acclerators=other") {
    edm::test::TestProcessor tester(config_other);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 2);
  }

  SECTION("Other disabled, accelerators=*") {
    edm::test::TestProcessor tester(configOtherDisabled_auto);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 1);
  }

  SECTION("Other disabled, accelerators=cpu") {
    edm::test::TestProcessor tester(configOtherDisabled_cpu);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 1);
  }
}
