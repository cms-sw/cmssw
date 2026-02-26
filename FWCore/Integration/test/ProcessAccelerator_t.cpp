#define CATCH_CONFIG_MAIN
#include "catch2/catch_all.hpp"

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "FWCore/TestProcessor/interface/TestProcessor.h"

#include <format>

#include <iostream>
#include <string_view>

static constexpr auto s_tag = "[ProcessAccelerator]";

namespace {
  std::string makeResolverConfig(bool enableOther, std::string_view accelerator, std::string_view variant) {
    const std::string appendOther = enableOther ? "self._enabled.append('other')" : "";
    const std::string explicitVariant =
        variant.empty() ? std::string(variant) : std::format(", variant=cms.untracked.string('{}')", variant);
    return std::format(
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
                        Catch::Matchers::ContainsSubstring("UnavailableAccelerator"));
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
