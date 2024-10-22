#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/TestProcessor/interface/TestProcessor.h"

#include <iostream>
#include <fmt/format.h>

static constexpr auto s_tag = "[EDAlias]";

TEST_CASE("Configuration", s_tag) {
  const std::string baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

process = TestProcess()

process.intprod = cms.EDProducer('IntProducer', ivalue = cms.int32(1))
process.intalias = cms.EDAlias(intprod = cms.VPSet(cms.PSet(type = cms.string('edmtestIntProduct'))))

# Module to be tested can not be an EDAlias
process.test = cms.EDProducer('AddIntsProducer', labels = cms.VInputTag('intalias'))

process.moduleToTest(process.test, cms.Task(process.intprod))
)_"};

  edm::test::TestProcessor::Config config{baseConfig};

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

  SECTION("Getting value") {
    edm::test::TestProcessor tester(config);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 1);
  }
}

TEST_CASE("Configuration with two instance aliases to a single product", s_tag) {
  const std::string baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

process = TestProcess()

process.intprod = cms.EDProducer('IntProducer', ivalue = cms.int32(1))
process.intalias = cms.EDAlias(intprod = cms.VPSet(cms.PSet(type = cms.string('edmtestIntProduct')),
                                                   cms.PSet(type = cms.string('edmtestIntProduct'),
                                                            fromProductInstance = cms.string(''),
                                                            toProductInstance = cms.string('bar'))
                                                  )
)

# Module to be tested can not be an EDAlias
process.test = cms.EDProducer('AddIntsProducer', labels = cms.VInputTag('intalias', 'intalias:bar'))

process.moduleToTest(process.test, cms.Task(process.intprod))
)_"};

  edm::test::TestProcessor::Config config{baseConfig};

  SECTION("Alias with two instances pointing to the same product is not allowed") {
    REQUIRE_THROWS_WITH(
        edm::test::TestProcessor(config),
        Catch::Contains("EDAlias conflict") && Catch::Contains("is used for multiple products of type") &&
            Catch::Contains("with module label") && Catch::Contains("and instance name") &&
            Catch::Contains("alias has the instance name") && Catch::Contains("and the other has the instance name"));
  }
}

TEST_CASE("Configuration with two identical aliases pointing to different products") {
  const std::string baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

process = TestProcess()

process.intprod = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2), values = cms.VPSet(cms.PSet(instance=cms.string('foo'),value=cms.int32(3))))
process.intalias = cms.EDAlias(intprod = cms.VPSet(cms.PSet(type = cms.string('edmtestIntProduct'), toProductInstance = cms.string(''), fromProductInstance = cms.string('')),
                                                   cms.PSet(type = cms.string('edmtestIntProduct'), toProductInstance = cms.string(''), fromProductInstance = cms.string('foo'))
                                                  )
)

# Module to be tested can not be an EDAlias
process.test = cms.EDProducer('AddIntsProducer', labels = cms.VInputTag('intalias'))

process.moduleToTest(process.test, cms.Task(process.intprod))
)_"};

  edm::test::TestProcessor::Config config{baseConfig};

  SECTION("Alias with two instances pointing to the same product is not allowed") {
    REQUIRE_THROWS_WITH(edm::test::TestProcessor(config),
                        Catch::Contains("EDAlias conflict") && Catch::Contains("and product instance alias") &&
                            Catch::Contains("are used for multiple products of type") &&
                            Catch::Contains("One has module label") && Catch::Contains("the other has module label"));
  }
}

////////////////////////////////////////
TEST_CASE("Configuration with all products of a module", s_tag) {
  const std::string baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

process = TestProcess()

process.intprod = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2),
    values = cms.VPSet(
        cms.PSet(instance=cms.string('foo'),value=cms.int32(3)),
        cms.PSet(instance=cms.string('another'),value=cms.int32(4)),
    )
)

process.intalias = cms.EDAlias(intprod = cms.EDAlias.allProducts())

# Module to be tested can not be an EDAlias
process.test = cms.EDProducer('AddIntsProducer',
    labels = cms.VInputTag('intalias', ('intalias', 'foo'), ('intalias', 'another'))
)

process.moduleToTest(process.test, cms.Task(process.intprod))
)_"};

  edm::test::TestProcessor::Config config{baseConfig};

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

  SECTION("Getting value") {
    edm::test::TestProcessor tester(config);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 9);
  }
}

TEST_CASE("Configuration with all products of a module with a given product instance name", s_tag) {
  constexpr const std::string_view baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

process = TestProcess()

process.intprod = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2),
    values = cms.VPSet(
        cms.PSet(instance=cms.string('foo'),value=cms.int32(3)),
        cms.PSet(instance=cms.string('another'),value=cms.int32(4)),
    )
)

process.intalias = cms.EDAlias(intprod = cms.VPSet(cms.PSet(type = cms.string('*'), fromProductInstance = cms.string('another'))))

# Module to be tested can not be an EDAlias
process.test = cms.EDProducer('AddIntsProducer',
    labels = cms.VInputTag({})
)

process.moduleToTest(process.test, cms.Task(process.intprod))
)_"};

  edm::test::TestProcessor::Config config{fmt::format(baseConfig, "'intalias:another'")};

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

  SECTION("Getting value") {
    edm::test::TestProcessor tester(config);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 4);
  }

  SECTION("Other product instances are not aliased") {
    {
      edm::test::TestProcessor::Config config{fmt::format(baseConfig, "'intalias:foo'")};
      edm::test::TestProcessor tester(config);
      REQUIRE_THROWS_WITH(tester.test(), Catch::Contains("ProductNotFound"));
    }
    {
      edm::test::TestProcessor::Config config{fmt::format(baseConfig, "'intalias'")};
      edm::test::TestProcessor tester(config);
      REQUIRE_THROWS_WITH(tester.test(), Catch::Contains("ProductNotFound"));
    }
  }
}

////////////////////////////////////////
TEST_CASE("Configuration with all products of two modules", s_tag) {
  const std::string baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

process = TestProcess()

process.intprod = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2),
    values = cms.VPSet(
        cms.PSet(instance=cms.string('foo'),value=cms.int32(3)),
        cms.PSet(instance=cms.string('another'),value=cms.int32(4)),
    )
)
process.intprod2 = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(20),
    values = cms.VPSet(
        cms.PSet(instance=cms.string('foo2'),value=cms.int32(30)),
        cms.PSet(instance=cms.string('another2'),value=cms.int32(40)),
    )
)

process.intalias = cms.EDAlias(
    intprod = cms.EDAlias.allProducts(),
    # can't use allProducts() because the product instance '' would lead to duplicate brances to be aliased
    intprod2 = cms.VPSet(
        cms.PSet(type = cms.string('*'), fromProductInstance = cms.string('foo2')),
        cms.PSet(type = cms.string('*'), fromProductInstance = cms.string('another2')),
    )
)

# Module to be tested can not be an EDAlias
process.test = cms.EDProducer('AddIntsProducer',
    labels = cms.VInputTag('intalias', ('intalias', 'foo'), ('intalias', 'another'), ('intalias', 'foo2'), ('intalias', 'another2'))
)

process.moduleToTest(process.test, cms.Task(process.intprod, process.intprod2))
)_"};

  edm::test::TestProcessor::Config config{baseConfig};

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

  SECTION("Getting value") {
    edm::test::TestProcessor tester(config);
    auto event = tester.test();
    REQUIRE(event.get<edmtest::IntProduct>()->value == 79);
  }
}

////////////////////////////////////////
TEST_CASE("No products found with wildcards", s_tag) {
  constexpr const std::string_view baseConfig{
      R"_(from FWCore.TestProcessor.TestProcess import *
import FWCore.ParameterSet.Config as cms

process = TestProcess()

process.intprod = cms.EDProducer('ManyIntProducer', ivalue = cms.int32(2),
    values = cms.VPSet(
        cms.PSet(instance=cms.string('foo'),value=cms.int32(3)),
        cms.PSet(instance=cms.string('another'),value=cms.int32(4)),
    )
)

process.intalias = cms.EDAlias({})

# Module to be tested can not be an EDAlias
process.test = cms.EDProducer('AddIntsProducer',
    labels = cms.VInputTag('intalias')
)

process.moduleToTest(process.test, cms.Task(process.intprod))
)_"};

  SECTION("Type wildcard") {
    edm::test::TestProcessor::Config config{fmt::format(
        baseConfig,
        "intprod = cms.VPSet(cms.PSet(type = cms.string('*'), fromProductInstance = cms.string('nonexistent')))")};

    REQUIRE_THROWS_WITH(
        edm::test::TestProcessor(config),
        Catch::Contains("There are no products with module label 'intprod' and product instance name 'nonexistent'"));
  }

  SECTION("Instance wildcard") {
    edm::test::TestProcessor::Config config{
        fmt::format(baseConfig, "intprod = cms.VPSet(cms.PSet(type = cms.string('nonexistentType')))")};

    REQUIRE_THROWS_WITH(edm::test::TestProcessor(config),
                        Catch::Contains("There are no products of type 'nonexistentType'") &&
                            Catch::Contains("with module label 'intprod'"));
  }
}
