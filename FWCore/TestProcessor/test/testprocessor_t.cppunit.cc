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
  
  (void) tester.test();
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>

