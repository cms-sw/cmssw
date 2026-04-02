/*
 *  tsetprocessor_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 5/1/18.
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/TestProcessor/interface/TestProcessor.h"

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"

#include "FWCore/Integration/interface/ESTestData.h"
#include "FWCore/Integration/interface/ESTestRecords.h"

#include "catch2/catch_all.hpp"

#include <memory>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <set>

TEST_CASE("TestProcessor tests", "[TestProcessor]") {
  SECTION("simple process test") {
    char const* kTest =
        "from FWCore.TestProcessor.TestProcess import *\n"
        "process = TestProcess()\n"
        "process.foo = cms.EDProducer('IntProducer', ivalue=cms.int32(1))\n"
        "process.moduleToTest(process.foo)\n";
    edm::test::TestProcessor::Config config(kTest);
    edm::test::TestProcessor tester(config);
    REQUIRE(tester.labelOfTestModule() == "foo");

    auto event = tester.test();

    REQUIRE(event.get<edmtest::IntProduct>()->value == 1);

    REQUIRE(not event.get<edmtest::IntProduct>("doesNotExist"));
    REQUIRE_THROWS_AS(*event.get<edmtest::IntProduct>("doesNotExist"), cms::Exception);
  }

  SECTION("add product test") {
    char const* kTest =
        "from FWCore.TestProcessor.TestProcess import *\n"
        "process = TestProcess()\n"
        "process.add = cms.EDProducer('AddIntsProducer', labels=cms.VInputTag('in'))\n"
        "process.moduleToTest(process.add)\n";
    edm::test::TestProcessor::Config config(kTest);
    auto token = config.produces<edmtest::IntProduct>("in");

    edm::test::TestProcessor tester(config);

    {
      auto event = tester.test(std::make_pair(token, std::make_unique<edmtest::IntProduct>(1)));

      REQUIRE(event.get<edmtest::IntProduct>()->value == 1);
    }

    {
      auto event = tester.test(std::make_pair(token, std::make_unique<edmtest::IntProduct>(2)));

      REQUIRE(event.get<edmtest::IntProduct>()->value == 2);
    }

    //Check that event gets reset so the data product is not available
    REQUIRE_THROWS_AS(tester.test(), cms::Exception);
  }

  SECTION("missing product test") {
    char const* kTest =
        "from FWCore.TestProcessor.TestProcess import *\n"
        "process = TestProcess()\n"
        "process.add = cms.EDProducer('AddIntsProducer', labels=cms.VInputTag('in'))\n"
        "process.moduleToTest(process.add)\n";
    edm::test::TestProcessor::Config config(kTest);

    edm::test::TestProcessor tester(config);

    REQUIRE_THROWS_AS(tester.test(), cms::Exception);
  }

  SECTION("filter test") {
    char const* kTest =
        "from FWCore.TestProcessor.TestProcess import *\n"
        "process = TestProcess()\n"
        "process.foo = cms.EDFilter('TestFilterModule', acceptValue=cms.untracked.int32(2),\n"
        "   onlyOne = cms.untracked.bool(True))\n"
        "process.moduleToTest(process.foo)\n";
    edm::test::TestProcessor::Config config(kTest);
    edm::test::TestProcessor tester(config);
    REQUIRE(tester.labelOfTestModule() == "foo");

    REQUIRE(not tester.test().modulePassed());
    REQUIRE(tester.test().modulePassed());
    REQUIRE(not tester.test().modulePassed());
    REQUIRE(tester.test().modulePassed());
  }

  SECTION("output module test") {
    char const* kTest =
        "from FWCore.TestProcessor.TestProcess import *\n"
        "process = TestProcess()\n"
        "process.MessageLogger.cerr.INFO.limit=10000\n"
        "process.foo = cms.OutputModule('GetProductCheckerOutputModule', verbose=cms.untracked.bool(True),"
        " outputCommands = cms.untracked.vstring('drop *','keep edmtestIntProduct_in__TEST'),\n"
        " crosscheck = cms.untracked.vstring('edmtestIntProduct_in__TEST'))\n"
        "process.moduleToTest(process.foo)\n";
    edm::test::TestProcessor::Config config(kTest);
    auto token = config.produces<edmtest::IntProduct>("in");
    edm::test::TestProcessor tester(config);
    tester.test(std::make_pair(token, std::make_unique<edmtest::IntProduct>(1)));
  }

  SECTION("extra process test") {
    char const* kTest =
        "from FWCore.TestProcessor.TestProcess import *\n"
        "process = TestProcess()\n"
        "process.add = cms.EDProducer('AddIntsProducer', labels=cms.VInputTag('in'))\n"
        "process.moduleToTest(process.add)\n";
    edm::test::TestProcessor::Config config(kTest);
    auto processToken = config.addExtraProcess("HLT");
    auto token = config.produces<edmtest::IntProduct>("in", "", processToken);

    edm::test::TestProcessor tester(config);

    {
      auto event = tester.test(std::make_pair(token, std::make_unique<edmtest::IntProduct>(1)));

      REQUIRE(event.get<edmtest::IntProduct>()->value == 1);
    }
  }

  SECTION("event setup test") {
    char const* kTest =
        "from FWCore.TestProcessor.TestProcess import *\n"
        "process = TestProcess()\n"
        "process.emptyESSourceA1 = cms.ESSource('EmptyESSource',"
        "recordName = cms.string('ESTestRecordA'),"
        "firstValid = cms.vuint32(1,2),"
        "iovIsRunNotTime = cms.bool(True)"
        ")\n"

        "process.add_(cms.ESProducer('ESTestProducerA') )\n"
        "process.add = cms.EDAnalyzer('ESTestAnalyzerA', runsToGetDataFor = cms.vint32(1,2), "
        "expectedValues=cms.untracked.vint32(1,2))\n"
        "process.moduleToTest(process.add)\n";
    edm::test::TestProcessor::Config config(kTest);

    edm::test::TestProcessor tester(config);

    (void)tester.test();

    tester.setRunNumber(2);
    (void)tester.test();
  }

  SECTION("event setup put test") {
    char const* kTest =
        "from FWCore.TestProcessor.TestProcess import *\n"
        "process = TestProcess()\n"
        "process.add = cms.EDAnalyzer('ESTestAnalyzerA', runsToGetDataFor = cms.vint32(1,2,3), "
        "expectedValues=cms.untracked.vint32(1,2,2))\n"
        "process.moduleToTest(process.add)\n";
    edm::test::TestProcessor::Config config(kTest);
    auto estoken = config.esProduces<ESTestRecordA, edmtest::ESTestDataA>();

    edm::test::TestProcessor tester(config);

    (void)tester.test(std::make_pair(estoken, std::make_unique<edmtest::ESTestDataA>(1)));

    tester.setRunNumber(2);
    (void)tester.test(std::make_pair(estoken, std::make_unique<edmtest::ESTestDataA>(2)));

    tester.setRunNumber(3);
    REQUIRE_THROWS_AS(tester.test(), cms::Exception);
  }

  SECTION("lumi test") {
    auto const kTest = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer('ThingProducer')
process.moduleToTest(process.toTest)
)_";
    edm::test::TestProcessor::Config config(kTest);

    edm::test::TestProcessor tester(config);
    (void)tester.test();
    tester.setLuminosityBlockNumber(2);
    (void)tester.test();
  }

  SECTION("task test") {
    char const* kTest =
        "from FWCore.TestProcessor.TestProcess import *\n"
        "process = TestProcess()\n"
        "process.mid = cms.EDProducer('AddIntsProducer', labels=cms.VInputTag('in'))\n"
        "process.add = cms.EDProducer('AddIntsProducer', labels=cms.VInputTag('mid','in'))\n"
        "process.moduleToTest(process.add,cms.Task(process.mid))\n";
    edm::test::TestProcessor::Config config(kTest);
    auto token = config.produces<edmtest::IntProduct>("in");

    edm::test::TestProcessor tester(config);

    {
      auto event = tester.test(std::make_pair(token, std::make_unique<edmtest::IntProduct>(1)));

      REQUIRE(event.get<edmtest::IntProduct>()->value == 2);
    }
  }

  SECTION("empty process block test") {
    auto const kTest = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDAnalyzer('RunLumiEventChecker',
        eventSequence = cms.untracked.VEventID()
                                )
process.moduleToTest(process.toTest)
)_";
    edm::test::TestProcessor::Config config(kTest);
    edm::test::TestProcessor tester(config);

    tester.testWithNoRuns();
  }

  SECTION("empty run test") {
    auto const kTest = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDAnalyzer('RunLumiEventChecker',
        eventSequence = cms.untracked.VEventID(cms.EventID(1,0,0), cms.EventID(1,0,0))
                                )
process.moduleToTest(process.toTest)
)_";
    edm::test::TestProcessor::Config config(kTest);

    edm::test::TestProcessor tester(config);

    tester.testRunWithNoLuminosityBlocks();
  }

  SECTION("empty lumi test") {
    auto const kTest = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDAnalyzer('RunLumiEventChecker',
                                  eventSequence = cms.untracked.VEventID(cms.EventID(1,0,0), cms.EventID(1,1,0),
                                                                         cms.EventID(1,1,0), cms.EventID(1,0,0))
                                  )
process.moduleToTest(process.toTest)
)_";
    edm::test::TestProcessor::Config config(kTest);

    edm::test::TestProcessor tester(config);

    tester.testLuminosityBlockWithNoEvents();
  }

  SECTION("process block product test") {
    auto const kTest = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.intProducerBeginProcessBlock = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(10000))
process.moduleToTest(process.intProducerBeginProcessBlock)
)_";
    edm::test::TestProcessor::Config config(kTest);

    edm::test::TestProcessor tester(config);
    {
      auto processBlock = tester.testBeginProcessBlock();
      REQUIRE(processBlock.get<edmtest::IntProduct>()->value == 10000);
    }
    {
      auto processBlock = tester.testEndProcessBlock();
      REQUIRE(processBlock.get<edmtest::IntProduct>()->value == 10000);
    }
  }

  SECTION("process block end product test") {
    auto const kTest = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.intProducerEndProcessBlock = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(10001))
process.moduleToTest(process.intProducerEndProcessBlock)
)_";
    edm::test::TestProcessor::Config config(kTest);

    edm::test::TestProcessor tester(config);
    {
      auto processBlock = tester.testBeginProcessBlock();
      REQUIRE_THROWS_AS(*processBlock.get<edmtest::IntProduct>(), cms::Exception);
    }
    {
      auto processBlock = tester.testEndProcessBlock();
      REQUIRE(processBlock.get<edmtest::IntProduct>()->value == 10001);
    }
  }

  SECTION("run product test") {
    auto const kTest = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer('ThingProducer')
process.moduleToTest(process.toTest)
)_";
    edm::test::TestProcessor::Config config(kTest);

    edm::test::TestProcessor tester(config);
    {
      auto run = tester.testBeginRun(1);
      REQUIRE(run.get<edmtest::ThingCollection>("beginRun")->size() == 20);
    }

    {
      auto run = tester.testEndRun();
      REQUIRE(run.get<edmtest::ThingCollection>("beginRun")->size() == 20);
      REQUIRE(run.get<edmtest::ThingCollection>("endRun")->size() == 20);
    }

    {
      auto run = tester.testBeginRun(2);
      REQUIRE(run.get<edmtest::ThingCollection>("beginRun")->size() == 20);
    }
  }

  SECTION("lumi product test") {
    auto const kTest = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer('ThingProducer')
process.moduleToTest(process.toTest)
)_";
    edm::test::TestProcessor::Config config(kTest);

    edm::test::TestProcessor tester(config);
    {
      auto lumi = tester.testBeginLuminosityBlock(1);
      REQUIRE(lumi.get<edmtest::ThingCollection>("beginLumi")->size() == 20);
    }

    {
      auto lumi = tester.testEndLuminosityBlock();
      REQUIRE(lumi.get<edmtest::ThingCollection>("beginLumi")->size() == 20);
      REQUIRE(lumi.get<edmtest::ThingCollection>("endLumi")->size() == 20);
    }

    {
      auto lumi = tester.testBeginLuminosityBlock(2);
      REQUIRE(lumi.get<edmtest::ThingCollection>("beginLumi")->size() == 20);
    }
  }

  SECTION("run stream analyzer test") {
    auto const kTest = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDAnalyzer('edmtest::stream::RunIntAnalyzer',
    transitions = cms.int32(2)
    ,cachevalue = cms.int32(0)
)

process.moduleToTest(process.toTest)
)_";
    edm::test::TestProcessor::Config config(kTest);
    edm::test::TestProcessor tester(config);
    tester.testBeginLuminosityBlock(1);
    tester.testEndLuminosityBlock();
    tester.testBeginLuminosityBlock(2);
  }

  SECTION("lumi stream analyzer test") {
    auto const kTest = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDAnalyzer('edmtest::stream::LumiIntAnalyzer',
    transitions = cms.int32(4)
    ,cachevalue = cms.int32(0)
    ,moduleLabel = cms.InputTag("")
)

process.moduleToTest(process.toTest)
)_";
    edm::test::TestProcessor::Config config(kTest);
    edm::test::TestProcessor tester(config);
    tester.testBeginLuminosityBlock(1);
    tester.testEndLuminosityBlock();
    tester.testBeginLuminosityBlock(2);
  }
}
