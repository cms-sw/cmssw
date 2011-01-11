/*
 *  makeprocess_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 5/18/05.
 *  Changed by Viji Sundararajan on 8-Jul-05.
 * 
 */


#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"

#include "boost/shared_ptr.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class testreadpsetsfrom: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testreadpsetsfrom);
CPPUNIT_TEST(simpleTest);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  void simpleTest();
private:

};



///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testreadpsetsfrom);

void testreadpsetsfrom::simpleTest()
{
   const char* kTest ="import FWCore.ParameterSet.Config as cms\n"
                      "dummy =  cms.PSet(b = cms.bool(True))\n"
                      "foo = cms.PSet(a = cms.string('blah'))\n"
   ;
   boost::shared_ptr<edm::ParameterSet> test = edm::readPSetsFrom(kTest);
   
   CPPUNIT_ASSERT(test->getParameterSet("dummy").getParameter<bool>("b")==true);
   CPPUNIT_ASSERT(test->getParameterSet("foo").getParameter<std::string>("a")==std::string("blah"));
}

