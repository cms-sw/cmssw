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

#include <cppunit/extensions/HelperMacros.h>

#include <memory>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <set>

class testTestProcessor : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testTestProcessor);
  CPPUNIT_TEST(simpleProcessTest);
  CPPUNIT_TEST(addProductTest);
  CPPUNIT_TEST(missingProductTest);
  CPPUNIT_TEST(filterTest);
  CPPUNIT_TEST(extraProcessTest);
  CPPUNIT_TEST(eventSetupTest);
  CPPUNIT_TEST(eventSetupPutTest);
  CPPUNIT_TEST(lumiTest);
  CPPUNIT_TEST(taskTest);
  CPPUNIT_TEST(emptyRunTest);
  CPPUNIT_TEST(emptyLumiTest);
  CPPUNIT_TEST(runProductTest);
  CPPUNIT_TEST(lumiProductTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void simpleProcessTest();
  void addProductTest();
  void missingProductTest();
  void filterTest();
  void extraProcessTest();
  void eventSetupTest();
  void eventSetupPutTest();
  void lumiTest();
  void taskTest();
  void emptyRunTest();
  void emptyLumiTest();
  void runProductTest();
  void lumiProductTest();

private:
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testTestProcessor);

void testTestProcessor::simpleProcessTest() {
  char const* kTest =
      "from FWCore.TestProcessor.TestProcess import *\n"
      "process = TestProcess()\n"
      "process.foo = cms.EDProducer('IntProducer', ivalue=cms.int32(1))\n"
      "process.moduleToTest(process.foo)\n";
  edm::test::TestProcessor::Config config(kTest);
  edm::test::TestProcessor tester(config);
  CPPUNIT_ASSERT(tester.labelOfTestModule() == "foo");

  auto event = tester.test();

  CPPUNIT_ASSERT(event.get<edmtest::IntProduct>()->value == 1);

  CPPUNIT_ASSERT(not event.get<edmtest::IntProduct>("doesNotExist"));
  CPPUNIT_ASSERT_THROW(*event.get<edmtest::IntProduct>("doesNotExist"), cms::Exception);
}

void testTestProcessor::addProductTest() {
  char const* kTest =
      "from FWCore.TestProcessor.TestProcess import *\n"
      "process = TestProcess()\n"
      "process.add = cms.EDProducer('AddIntsProducer', labels=cms.vstring('in'))\n"
      "process.moduleToTest(process.add)\n";
  edm::test::TestProcessor::Config config(kTest);
  auto token = config.produces<edmtest::IntProduct>("in");

  edm::test::TestProcessor tester(config);

  {
    auto event = tester.test(std::make_pair(token, std::make_unique<edmtest::IntProduct>(1)));

    CPPUNIT_ASSERT(event.get<edmtest::IntProduct>()->value == 1);
  }

  {
    auto event = tester.test(std::make_pair(token, std::make_unique<edmtest::IntProduct>(2)));

    CPPUNIT_ASSERT(event.get<edmtest::IntProduct>()->value == 2);
  }

  //Check that event gets reset so the data product is not available
  CPPUNIT_ASSERT_THROW(tester.test(), cms::Exception);
}

void testTestProcessor::missingProductTest() {
  char const* kTest =
      "from FWCore.TestProcessor.TestProcess import *\n"
      "process = TestProcess()\n"
      "process.add = cms.EDProducer('AddIntsProducer', labels=cms.vstring('in'))\n"
      "process.moduleToTest(process.add)\n";
  edm::test::TestProcessor::Config config(kTest);

  edm::test::TestProcessor tester(config);

  CPPUNIT_ASSERT_THROW(tester.test(), cms::Exception);
}

void testTestProcessor::filterTest() {
  char const* kTest =
      "from FWCore.TestProcessor.TestProcess import *\n"
      "process = TestProcess()\n"
      "process.foo = cms.EDFilter('TestFilterModule', acceptValue=cms.untracked.int32(2),\n"
      "   onlyOne = cms.untracked.bool(True))\n"
      "process.moduleToTest(process.foo)\n";
  edm::test::TestProcessor::Config config(kTest);
  edm::test::TestProcessor tester(config);
  CPPUNIT_ASSERT(tester.labelOfTestModule() == "foo");

  CPPUNIT_ASSERT(not tester.test().modulePassed());
  CPPUNIT_ASSERT(tester.test().modulePassed());
  CPPUNIT_ASSERT(not tester.test().modulePassed());
  CPPUNIT_ASSERT(tester.test().modulePassed());
}

void testTestProcessor::extraProcessTest() {
  char const* kTest =
      "from FWCore.TestProcessor.TestProcess import *\n"
      "process = TestProcess()\n"
      "process.add = cms.EDProducer('AddIntsProducer', labels=cms.vstring('in'))\n"
      "process.moduleToTest(process.add)\n";
  edm::test::TestProcessor::Config config(kTest);
  auto processToken = config.addExtraProcess("HLT");
  auto token = config.produces<edmtest::IntProduct>("in", "", processToken);

  edm::test::TestProcessor tester(config);

  {
    auto event = tester.test(std::make_pair(token, std::make_unique<edmtest::IntProduct>(1)));

    CPPUNIT_ASSERT(event.get<edmtest::IntProduct>()->value == 1);
  }
}

void testTestProcessor::eventSetupTest() {
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

void testTestProcessor::eventSetupPutTest() {
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
  CPPUNIT_ASSERT_THROW(tester.test(), cms::Exception);
}

void testTestProcessor::lumiTest() {
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

void testTestProcessor::taskTest() {
  char const* kTest =
      "from FWCore.TestProcessor.TestProcess import *\n"
      "process = TestProcess()\n"
      "process.mid = cms.EDProducer('AddIntsProducer', labels=cms.vstring('in'))\n"
      "process.add = cms.EDProducer('AddIntsProducer', labels=cms.vstring('mid','in'))\n"
      "process.moduleToTest(process.add,cms.Task(process.mid))\n";
  edm::test::TestProcessor::Config config(kTest);
  auto token = config.produces<edmtest::IntProduct>("in");

  edm::test::TestProcessor tester(config);

  {
    auto event = tester.test(std::make_pair(token, std::make_unique<edmtest::IntProduct>(1)));

    CPPUNIT_ASSERT(event.get<edmtest::IntProduct>()->value == 2);
  }
}

void testTestProcessor::emptyRunTest() {
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
void testTestProcessor::emptyLumiTest() {
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

void testTestProcessor::runProductTest() {
  auto const kTest = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer('ThingProducer')
process.moduleToTest(process.toTest)
)_";
  edm::test::TestProcessor::Config config(kTest);

  edm::test::TestProcessor tester(config);
  {
    auto run = tester.testBeginRun(1);
    CPPUNIT_ASSERT(run.get<edmtest::ThingCollection>("beginRun")->size() == 20);
  }

  {
    auto run = tester.testEndRun();
    CPPUNIT_ASSERT(run.get<edmtest::ThingCollection>("beginRun")->size() == 20);
    CPPUNIT_ASSERT(run.get<edmtest::ThingCollection>("endRun")->size() == 20);
  }

  {
    auto run = tester.testBeginRun(2);
    CPPUNIT_ASSERT(run.get<edmtest::ThingCollection>("beginRun")->size() == 20);
  }
}
void testTestProcessor::lumiProductTest() {
  auto const kTest = R"_(from FWCore.TestProcessor.TestProcess import *
process = TestProcess()
process.toTest = cms.EDProducer('ThingProducer')
process.moduleToTest(process.toTest)
)_";
  edm::test::TestProcessor::Config config(kTest);

  edm::test::TestProcessor tester(config);
  {
    auto lumi = tester.testBeginLuminosityBlock(1);
    CPPUNIT_ASSERT(lumi.get<edmtest::ThingCollection>("beginLumi")->size() == 20);
  }

  {
    auto lumi = tester.testEndLuminosityBlock();
    CPPUNIT_ASSERT(lumi.get<edmtest::ThingCollection>("beginLumi")->size() == 20);
    CPPUNIT_ASSERT(lumi.get<edmtest::ThingCollection>("endLumi")->size() == 20);
  }

  {
    auto lumi = tester.testBeginLuminosityBlock(2);
    CPPUNIT_ASSERT(lumi.get<edmtest::ThingCollection>("beginLumi")->size() == 20);
  }
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
