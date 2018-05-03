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
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  void simpleProcessTest();
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

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>

