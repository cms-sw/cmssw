#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "FWCore/TestProcessor/interface/TestProcessor.h"

#include <iostream>

static constexpr auto s_tag = "[SwitchProducer]";

namespace {
  std::string makeConfig(bool test2Enabled,
                         const std::string& test1,
                         const std::string& test2,
                         const std::string& otherprod = "",
                         const std::string& othername = "") {
    std::string otherline;
    std::string othertask;
    if (not otherprod.empty()) {
      otherline = "process." + othername + " = " + otherprod + "\n";
      othertask = ", cms.Task(process." + othername + ")";
    }

    return std::string{
               R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

class SwitchProducerTest(cms.SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerTest,self).__init__(
            dict(
                test1 = lambda accelerators: (True, -10),
                test2 = lambda accelerators: ()_"} +
           (test2Enabled ? "True" : "False") + ", -9)\n" +
           R"_(            ), **kargs)
process = TestProcess()
)_" + otherline +
           R"_(process.s = SwitchProducerTest(
   test1 = )_" +
           test1 + ",\n" + "   test2 = " + test2 + "\n" + ")\n" + "process.moduleToTest(process.s" + othertask + ")\n";
  }
}  // namespace

TEST_CASE("Configuration", s_tag) {
  const std::string test1{"cms.EDProducer('IntProducer', ivalue = cms.int32(1))"};
  const std::string test2{"cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet())"};

  const std::string baseConfig = makeConfig(true, test1, test2);
  const std::string baseConfigTest2Disabled = makeConfig(false, test1, test2);

  SECTION("Configuration hash is not changed") {
    auto pset = edm::readConfig(baseConfig);
    auto psetTest2Disabled = edm::readConfig(baseConfigTest2Disabled);
    pset->registerIt();
    psetTest2Disabled->registerIt();
    REQUIRE(pset->id() == psetTest2Disabled->id());
  }

  edm::test::TestProcessor::Config config{baseConfig};
  edm::test::TestProcessor::Config configTest2Disabled{baseConfigTest2Disabled};

  SECTION("Base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

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
}

TEST_CASE("Configuration with EDAlias", s_tag) {
  const std::string otherprod{"cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet())"};
  const std::string othername{"intprod"};

  const std::string test1{"cms.EDProducer('IntProducer', ivalue = cms.int32(1))"};
  const std::string test2{"cms.EDAlias(intprod = cms.VPSet(cms.PSet(type = cms.string('edmtestIntProduct'))))"};

  const std::string baseConfig = makeConfig(true, test1, test2, otherprod, othername);
  const std::string baseConfigTest2Disabled = makeConfig(false, test1, test2, otherprod, othername);

  SECTION("Configuration hash is not changed") {
    auto pset = edm::readConfig(baseConfig);
    auto psetTest2Disabled = edm::readConfig(baseConfigTest2Disabled);
    pset->registerIt();
    psetTest2Disabled->registerIt();
    REQUIRE(pset->id() == psetTest2Disabled->id());
  }

  edm::test::TestProcessor::Config config{baseConfig};
  edm::test::TestProcessor::Config configTest2Disabled{baseConfigTest2Disabled};

  SECTION("Base configuration is OK") { REQUIRE_NOTHROW(edm::test::TestProcessor(config)); }

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
}

TEST_CASE("Configuration with many branches", s_tag) {
  const std::string test1{
      R"_(cms.EDProducer('ManyIntProducer', ivalue = cms.int32(1), values = cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(11)),
                                                                                       cms.PSet(instance=cms.string('bar'),value=cms.int32(21)))))_"};
  const std::string test2{
      R"_(cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(12)),
                                                                                       cms.PSet(instance=cms.string('bar'),value=cms.int32(22)))))_"};

  const std::string baseConfig = makeConfig(true, test1, test2);
  const std::string baseConfigTest2Disabled = makeConfig(false, test1, test2);

  edm::test::TestProcessor::Config config{baseConfig};
  edm::test::TestProcessor::Config configTest2Disabled{baseConfigTest2Disabled};

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

TEST_CASE("Configuration with many branches with EDAlias", s_tag) {
  const std::string otherprod{
      R"_(cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(12)),
                                                                                       cms.PSet(instance=cms.string('bar'),value=cms.int32(22))))
)_"};
  const std::string othername{"intprod"};

  const std::string test1{
      R"_(cms.EDProducer('ManyIntProducer', ivalue = cms.int32(1), values = cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(11)),
                                                                                       cms.PSet(instance=cms.string('bar'),value=cms.int32(21)))))_"};
  const std::string test2{
      R"_(cms.EDAlias(intprod = cms.VPSet(cms.PSet(type = cms.string('edmtestIntProduct'), fromProductInstance = cms.string(''), toProductInstance = cms.string('')),
                                                              cms.PSet(type = cms.string('edmtestIntProduct'), fromProductInstance = cms.string('foo'), toProductInstance = cms.string('foo')),
                                                              cms.PSet(type = cms.string('edmtestIntProduct'), fromProductInstance = cms.string('bar'), toProductInstance = cms.string('bar')))))_"};

  const std::string baseConfig = makeConfig(true, test1, test2, otherprod, othername);
  const std::string baseConfigTest2Disabled = makeConfig(false, test1, test2, otherprod, othername);

  edm::test::TestProcessor::Config config{baseConfig};
  edm::test::TestProcessor::Config configTest2Disabled{baseConfigTest2Disabled};

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
  const std::string test1{"cms.EDProducer('ManyIntProducer', ivalue = cms.int32(1), values = cms.VPSet())"};
  const std::string test2{
      "cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = "
      "cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(3))))"};

  const std::string baseConfig1 = makeConfig(true, test1, test2);
  const std::string baseConfig2 = makeConfig(false, test1, test2);

  SECTION("Different branches are not allowed") {
    edm::test::TestProcessor::Config config1{baseConfig1};
    REQUIRE_THROWS_WITH(
        edm::test::TestProcessor(config1),
        Catch::Contains("that does not produce a product") && Catch::Contains("that is produced by the chosen case") &&
            Catch::Contains("Products for case s@test1") && Catch::Contains("Products for case s@test2") &&
            Catch::Contains("edmtestIntProduct \n") && Catch::Contains("edmtestIntProduct foo"));

    edm::test::TestProcessor::Config config2{baseConfig2};
    REQUIRE_THROWS_WITH(
        edm::test::TestProcessor(config2),
        Catch::Contains("with a product") && Catch::Contains("that is not produced by the chosen case") &&
            Catch::Contains("Products for case s@test1") && Catch::Contains("Products for case s@test2") &&
            Catch::Contains("edmtestIntProduct \n") && Catch::Contains("edmtestIntProduct foo"));
  }
}

TEST_CASE("Configuration with different transient branches", s_tag) {
  const std::string test1{"cms.EDProducer('ManyIntProducer', ivalue = cms.int32(1), values = cms.VPSet())"};
  const std::string test2{
      "cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), transientValues = "
      "cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(3))))"};

  const std::string baseConfig1 = makeConfig(true, test1, test2);
  const std::string baseConfig2 = makeConfig(false, test1, test2);

  SECTION("Different transient branches are allowed") {
    edm::test::TestProcessor::Config config1{baseConfig1};
    edm::test::TestProcessor testProcessor1{config1};
    auto event1 = testProcessor1.test();
    REQUIRE(event1.get<edmtest::IntProduct>()->value == 2);

    // It would be better if the next line of executable code would
    // throw an exception, but that is not the current expected behavior.
    // It would be nice to have all cases of a SwitchProducer fail if
    // any case fails on a "get" (but it is intentional and desirable
    // that it does not fail only because a  case declares it produces
    // different transient products). We decided it was not worth the
    // effort and complexity to implement this behavior (at least not yet).
    REQUIRE(event1.get<edmtest::TransientIntProduct>("foo")->value == 3);

    edm::test::TestProcessor::Config config2{baseConfig2};
    edm::test::TestProcessor testProcessor2{config2};
    auto event2 = testProcessor2.test();
    REQUIRE(event2.get<edmtest::IntProduct>()->value == 1);
    REQUIRE_THROWS(event2.get<edmtest::TransientIntProduct>()->value == 3);
  }
}

TEST_CASE("Configuration with different branches with EDAlias", s_tag) {
  const std::string otherprod{
      "cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = "
      "cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(3))))"};
  const std::string othername{"intprod"};

  const std::string test1{"cms.EDProducer('ManyIntProducer', ivalue = cms.int32(1), values = cms.VPSet())"};
  const std::string test2{
      R"_(cms.EDAlias(intprod = cms.VPSet(cms.PSet(type = cms.string('edmtestIntProduct'), fromProductInstance = cms.string(''), toProductInstance = cms.string('')),
                                                              cms.PSet(type = cms.string('edmtestIntProduct'), fromProductInstance = cms.string('foo'), toProductInstance = cms.string('foo')))))_"};

  const std::string baseConfig1 = makeConfig(true, test1, test2, otherprod, othername);
  const std::string baseConfig2 = makeConfig(false, test1, test2, otherprod, othername);

  SECTION("Different branches are not allowed") {
    edm::test::TestProcessor::Config config1{baseConfig1};
    REQUIRE_THROWS_WITH(
        edm::test::TestProcessor(config1),
        Catch::Contains("that does not produce a product") && Catch::Contains("that is produced by the chosen case") &&
            Catch::Contains("Products for case s@test1") && Catch::Contains("Products for case s@test2") &&
            Catch::Contains("edmtestIntProduct \n") && Catch::Contains("edmtestIntProduct foo"));

    edm::test::TestProcessor::Config config2{baseConfig2};
    REQUIRE_THROWS_WITH(
        edm::test::TestProcessor(config2),
        Catch::Contains("with a product") && Catch::Contains("that is not produced by the chosen case") &&
            Catch::Contains("Products for case s@test1") && Catch::Contains("Products for case s@test2") &&
            Catch::Contains("edmtestIntProduct \n") && Catch::Contains("edmtestIntProduct foo"));
  }
}

TEST_CASE("Configuration with lumi and run", s_tag) {
  const std::string test1{"cms.EDProducer('ThingProducer', nThings = cms.int32(10))"};
  const std::string test2{"cms.EDProducer('ThingProducer', nThings = cms.int32(20))"};
  const std::string baseConfig = makeConfig(true, test1, test2);

  edm::test::TestProcessor::Config config{baseConfig};

  SECTION("Lumi and run products are not supported") {
    REQUIRE_THROWS_WITH(edm::test::TestProcessor(config),
                        Catch::Contains("SwitchProducer does not support non-event branches"));
  }
};

TEST_CASE("Configuration with ROOT branch alias", s_tag) {
  const std::string test1{
      "cms.EDProducer('ManyIntProducer', ivalue = cms.int32(1), values = "
      "cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(3))))"};
  const std::string test2{
      "cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = "
      "cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(4),branchAlias=cms.string('bar'))))"};

  const std::string baseConfig1 = makeConfig(true, test1, test2);
  const std::string baseConfig2 = makeConfig(false, test1, test2);

  SECTION("ROOT branch aliases are not supported for the chosen case") {
    edm::test::TestProcessor::Config config{baseConfig1};
    REQUIRE_THROWS_WITH(edm::test::TestProcessor(config),
                        Catch::Contains("SwitchProducer does not support ROOT branch aliases"));
  }

  SECTION("ROOT branch aliases are not supported for the non-chosen case") {
    edm::test::TestProcessor::Config config{baseConfig2};
    REQUIRE_THROWS_WITH(edm::test::TestProcessor(config),
                        Catch::Contains("SwitchProducer does not support ROOT branch aliases"));
  }
}
