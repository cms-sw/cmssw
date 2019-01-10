#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
#include "FWCore/TestProcessor/interface/TestProcessor.h"

#include <iostream>

static constexpr auto s_tag = "[SwitchProducer]";

TEST_CASE("Configuration", s_tag) {
  const std::string baseConfig{
R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda: (True, -10),
                test2 = lambda: (True, -9)
            ), **kargs)

process = TestProcess()
process.s = SwitchProducerTest(
   test1 = cms.EDProducer('IntProducer', ivalue = cms.int32(1)),
   test2 = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet())
)
process.moduleToTest(process.s)
)_"};

  const std::string baseConfigTest2Disabled{
R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda: (True, -10),
                test2 = lambda: (False, -9)
            ), **kargs)

process = TestProcess()
process.s = SwitchProducerTest(
   test1 = cms.EDProducer('IntProducer', ivalue = cms.int32(1)),
   test2 = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet())
)
process.moduleToTest(process.s)
)_"};

  SECTION("Configuration hash is not changed") {
    auto pset = edm::boost_python::readConfig(baseConfig);
    auto psetTest2Disabled = edm::boost_python::readConfig(baseConfigTest2Disabled);
    pset->registerIt();
    psetTest2Disabled->registerIt();
    REQUIRE(pset->id() == psetTest2Disabled->id());
  }

  edm::test::TestProcessor::Config config{ baseConfig };
  edm::test::TestProcessor::Config configTest2Disabled{ baseConfigTest2Disabled };

  SECTION("Base configuration is OK") {
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

  SECTION("Test2 enabled") {
    edm::test::TestProcessor tester(config);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 2);
  }

  SECTION("Test2 disabled") {
    edm::test::TestProcessor tester(configTest2Disabled);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 1);
  }
};

TEST_CASE("Configuration with many branches", s_tag) {
  const std::string baseConfig{
R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda: (True, -10),
                test2 = lambda: (True, -9)
            ), **kargs)

process = TestProcess()
process.s = SwitchProducerTest(
   test1 = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(1), values = cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(11)),
                                                                                       cms.PSet(instance=cms.string('bar'),value=cms.int32(21)))),
   test2 = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(12)),
                                                                                       cms.PSet(instance=cms.string('bar'),value=cms.int32(22))))
)
process.moduleToTest(process.s)
)_"};
  const std::string baseConfigTest2Disabled{
R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda: (True, -10),
                test2 = lambda: (False, -9)
            ), **kargs)

process = TestProcess()
process.s = SwitchProducerTest(
   test1 = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(1), values = cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(11)),
                                                                                       cms.PSet(instance=cms.string('bar'),value=cms.int32(21)))),
   test2 = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(12)),
                                                                                       cms.PSet(instance=cms.string('bar'),value=cms.int32(22))))
)
process.moduleToTest(process.s)
)_"};

  edm::test::TestProcessor::Config config{ baseConfig };
  edm::test::TestProcessor::Config configTest2Disabled{ baseConfigTest2Disabled };

  SECTION("Test2 enabled") {
    edm::test::TestProcessor tester(config);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 2);
    REQUIRE(event.get<edmtest::IntProduct>("foo")->value == 12);
    REQUIRE(event.get<edmtest::IntProduct>("bar")->value == 22);
  }

  SECTION("Test2 disabled") {
    edm::test::TestProcessor tester(configTest2Disabled);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 1);
    REQUIRE(event.get<edmtest::IntProduct>("foo")->value == 11);
    REQUIRE(event.get<edmtest::IntProduct>("bar")->value == 21);
  }

}


TEST_CASE("Configuration with different branches", s_tag) {
  const std::string baseConfig1{
R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda: (True, -10),
                test2 = lambda: (True, -9)
            ), **kargs)

process = TestProcess()
process.s = SwitchProducerTest(
   test1 = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(1), values = cms.VPSet()),
   test2 = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(3))))
)
process.moduleToTest(process.s)
)_"};

    const std::string baseConfig2{
R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda: (True, -10),
                test2 = lambda: (True, -9)
            ), **kargs)

process = TestProcess()
process.s = SwitchProducerTest(
   test1 = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(1), values = cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(3)))),
   test2 = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet())
)
process.moduleToTest(process.s, cms.Task(process.i))
)_"};

  SECTION("Different branches are not allowed") {
    edm::test::TestProcessor::Config config1{ baseConfig1 };
    REQUIRE_THROWS_WITH(edm::test::TestProcessor(config1), Catch::Contains("that does not produce a product") && Catch::Contains("that is produced by the chosen case"));

    edm::test::TestProcessor::Config config2{ baseConfig2 };
    REQUIRE_THROWS(edm::test::TestProcessor(config1), Catch::Contains("with a product") && Catch::Contains("that is not produced by the chosen case"));
  }
}



TEST_CASE("Configuration with lumi and run", s_tag) {
  const std::string baseConfig{
R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda: (True, -10),
                test2 = lambda: (True, -9)
            ), **kargs)

process = TestProcess()
process.s = SwitchProducerTest(
   test1 = cms.EDProducer('ThingProducer', nThings = cms.int32(10)),
   test2 = cms.EDProducer('ThingProducer', nThings = cms.int32(20))
)
process.moduleToTest(process.s)
)_"};

  edm::test::TestProcessor::Config config{ baseConfig };

  SECTION("Lumi and run products are not supported") {
    REQUIRE_THROWS_WITH(edm::test::TestProcessor(config), Catch::Contains("SwitchProducer does not support non-event branches"));
  }
};


TEST_CASE("Configuration with ROOT branch alias", s_tag) {
  const std::string baseConfig1{
R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda: (True, -10),
                test2 = lambda: (True, -9)
            ), **kargs)

process = TestProcess()
process.s = SwitchProducerTest(
   test1 = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(1), values = cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(3)))),
   test2 = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(4),branchAlias=cms.string('bar'))))
)
process.moduleToTest(process.s)
)_"};

    const std::string baseConfig2{
R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda: (True, -10),
                test2 = lambda: (True, -9)
            ), **kargs)

process = TestProcess()
process.s = SwitchProducerTest(
   test1 = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(1), values = cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(3),branchAlias=cms.string('bar')))),
   test2 = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(4))))
)
process.moduleToTest(process.s)
)_"};

  SECTION("ROOT branch aliases are not supported for the chosen case") {
    edm::test::TestProcessor::Config config{ baseConfig1 };
    REQUIRE_THROWS_WITH(edm::test::TestProcessor(config), Catch::Contains("SwitchProducer does not support ROOT branch aliases"));
  }

  SECTION("ROOT branch aliases are not supported for the non-chosen case") {
    edm::test::TestProcessor::Config config{ baseConfig2 };
    REQUIRE_THROWS_WITH(edm::test::TestProcessor(config), Catch::Contains("SwitchProducer does not support ROOT branch aliases"));
  }
}
