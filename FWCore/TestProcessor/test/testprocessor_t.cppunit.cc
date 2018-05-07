/*
 *  tsetprocessor_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 5/1/18.
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "FWCore/PythonParameterSet/interface/PythonProcessDesc.h"

#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include <cppunit/extensions/HelperMacros.h>

#include <memory>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <set>

class testTestProcessor: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testTestProcessor);
  CPPUNIT_TEST(simpleProcessTest);
  CPPUNIT_TEST(addProductTest);
  CPPUNIT_TEST(missingProductTest);
  CPPUNIT_TEST(filterTest);
  CPPUNIT_TEST(extraProcessTest);
  CPPUNIT_TEST(eventSetupTest);

CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  void simpleProcessTest();
  void addProductTest();
  void missingProductTest();
  void filterTest();
  void extraProcessTest();
  void eventSetupTest();
private:

};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testTestProcessor);

void testTestProcessor::simpleProcessTest() {
   char const* kTest = "from FWCore.TestProcessor.TestProcess import *\n"
                       "process = TestProcess()\n"
                       "process.foo = cms.EDProducer('IntProducer', ivalue=cms.int32(1))\n"
                       "process.moduleToTest(process.foo)\n";
  edm::test::TestProcessor::Config config(kTest);
  edm::test::TestProcessor tester(config);
  CPPUNIT_ASSERT(tester.labelOfTestModule() == "foo");
  
  auto event=tester.test();
  
  CPPUNIT_ASSERT(event.get<edmtest::IntProduct>()->value == 1);

  CPPUNIT_ASSERT(not event.get<edmtest::IntProduct>("doesNotExist"));
  CPPUNIT_ASSERT_THROW( *event.get<edmtest::IntProduct>("doesNotExist"), cms::Exception);
}

void testTestProcessor::addProductTest() {
  char const* kTest = "from FWCore.TestProcessor.TestProcess import *\n"
  "process = TestProcess()\n"
  "process.add = cms.EDProducer('AddIntsProducer', labels=cms.vstring('in'))\n"
  "process.moduleToTest(process.add)\n";
  edm::test::TestProcessor::Config config(kTest);
  auto token = config.produces<edmtest::IntProduct>("in");

  edm::test::TestProcessor tester(config);

  {
    auto event=tester.test(std::make_pair(token,std::make_unique<edmtest::IntProduct>(1)));
  
    CPPUNIT_ASSERT(event.get<edmtest::IntProduct>()->value == 1);
  }
  
  {
    auto event=tester.test(std::make_pair(token,std::make_unique<edmtest::IntProduct>(2)));
    
    CPPUNIT_ASSERT(event.get<edmtest::IntProduct>()->value == 2);
  }

  //Check that event gets reset so the data product is not available
  CPPUNIT_ASSERT_THROW( tester.test(), cms::Exception);
  
}

void testTestProcessor::missingProductTest() {
  char const* kTest = "from FWCore.TestProcessor.TestProcess import *\n"
  "process = TestProcess()\n"
  "process.add = cms.EDProducer('AddIntsProducer', labels=cms.vstring('in'))\n"
  "process.moduleToTest(process.add)\n";
  edm::test::TestProcessor::Config config(kTest);
  
  edm::test::TestProcessor tester(config);
  
  CPPUNIT_ASSERT_THROW(tester.test(), cms::Exception);
  
}

void testTestProcessor::filterTest() {
  char const* kTest = "from FWCore.TestProcessor.TestProcess import *\n"
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
  char const* kTest = "from FWCore.TestProcessor.TestProcess import *\n"
  "process = TestProcess()\n"
  "process.add = cms.EDProducer('AddIntsProducer', labels=cms.vstring('in'))\n"
  "process.moduleToTest(process.add)\n";
  edm::test::TestProcessor::Config config(kTest);
  auto processToken = config.addExtraProcess("HLT");
  auto token = config.produces<edmtest::IntProduct>("in","",processToken);
  
  edm::test::TestProcessor tester(config);
  
  {
    auto event=tester.test(std::make_pair(token,std::make_unique<edmtest::IntProduct>(1)));
    
    CPPUNIT_ASSERT(event.get<edmtest::IntProduct>()->value == 1);
  }
  
}

void testTestProcessor::eventSetupTest() {
  char const* kTest = "from FWCore.TestProcessor.TestProcess import *\n"
  "process = TestProcess()\n"
  "process.emptyESSourceA1 = cms.ESSource('EmptyESSource',"
                                         "recordName = cms.string('ESTestRecordA'),"
                                         "firstValid = cms.vuint32(1,2),"
                                         "iovIsRunNotTime = cms.bool(True)"
                                         ")\n"

  "process.add_(cms.ESProducer('ESTestProducerA') )\n"
  "process.add = cms.EDAnalyzer('ESTestAnalyzerA', runsToGetDataFor = cms.vint32(1,2), expectedValues=cms.untracked.vint32(1,2))\n"
  "process.moduleToTest(process.add)\n";
  edm::test::TestProcessor::Config config(kTest);
  
  edm::test::TestProcessor tester(config);
  
  (void) tester.test();

  tester.setRunNumber(2);
  (void) tester.test();
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>

