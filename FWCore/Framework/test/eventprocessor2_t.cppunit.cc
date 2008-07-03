/*----------------------------------------------------------------------

Test of the EventProcessor class.

$Id: eventprocessor2_t.cppunit.cc,v 1.12 2008/04/04 16:11:03 wdd Exp $

----------------------------------------------------------------------*/  
#include <exception>
#include <iostream>
#include <string>
#include <stdexcept>
#include "FWCore/Framework/interface/EventProcessor.h"
#include <cppunit/extensions/HelperMacros.h>


class testeventprocessor2: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testeventprocessor2);
CPPUNIT_TEST(eventprocessor2Test);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  void eventprocessor2Test();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testeventprocessor2);

void work()
{
  std::string configuration(
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('PROD')\n"
      "process.maxEvents = cms.untracked.PSet(\n"
      "  input = cms.untracked.int32(5))\n"
      "process.source = cms.Source('EmptySource')\n"
      "process.m1 = cms.EDProducer('IntProducer',\n"
      "   ivalue = cms.int32(10))\n"
      "process.m2 = cms.EDProducer('DoubleProducer',\n"
      "   dvalue = cms.double(3.3))\n"
      "process.out = cms.OutputModule('AsciiOutputModule')\n"
      "process.p1 = cms.Path(process.m1*process.m2)\n"
      "process.ep1 = cms.EndPath(process.out)");
  edm::EventProcessor proc(configuration, true);
  proc.beginJob();
  proc.run();
  proc.endJob();
}

void testeventprocessor2::eventprocessor2Test()
{
  try { work();}
  catch (cms::Exception& e) {
      std::cerr << "CMS exception caught: "
		<< e.explainSelf() << std::endl;
      CPPUNIT_ASSERT("Exception caught in testeventprocessor2::eventprocessor2Test"==0);
  }
  catch (std::runtime_error& e) {
      std::cerr << "Standard library exception caught: "
		<< e.what() << std::endl;
      CPPUNIT_ASSERT("Exception caught in testeventprocessor2::eventprocessor2Test"==0);
  }
  catch (...) {
      std::cerr << "Unknown exception caught" << std::endl;
      CPPUNIT_ASSERT("Exception caught in testeventprocessor2::eventprocessor2Test"==0);
  }
}
