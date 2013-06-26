/*----------------------------------------------------------------------

Test of the EventProcessor class.

----------------------------------------------------------------------*/  
#include <exception>
#include <iostream>
#include <string>
#include <stdexcept>
#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"

// to be called also by the other cppunit...
void doInit() {
   static bool firstTime=true;
   if(firstTime) {
      //std::cout << "common init" << std::endl;
      edm::RootAutoLibraryLoader::enable();
      if(not edmplugin::PluginManager::isAvailable()) {
        edmplugin::PluginManager::configure(edmplugin::standard::config());
     }
      firstTime = false;
   }
}


class testeventprocessor2: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testeventprocessor2);
CPPUNIT_TEST(eventprocessor2Test);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){
      //std::cout << "setting up testeventprocessor2" << std::endl;
      doInit();
  }
  void tearDown(){}
  void eventprocessor2Test();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testeventprocessor2);



void work()
{
  //std::cout << "work in testeventprocessor2" << std::endl;
  std::string configuration(
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('PROD')\n"
      "process.maxEvents = cms.untracked.PSet(\n"
      "  input = cms.untracked.int32(5))\n"
      "process.source = cms.Source('EmptySource')\n"
      "process.m1 = cms.EDProducer('IntProducer',\n"
      "   ivalue = cms.int32(10))\n"
      "process.m2 = cms.EDProducer('ToyDoubleProducer',\n"
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
      CPPUNIT_ASSERT("cms Exception caught in testeventprocessor2::eventprocessor2Test"==0);
  }
  catch (std::runtime_error& e) {
      std::cerr << "Standard library exception caught: "
		<< e.what() << std::endl;
      CPPUNIT_ASSERT("std Exception caught in testeventprocessor2::eventprocessor2Test"==0);
  }
  catch (...) {
      std::cerr << "Unknown exception caught" << std::endl;
      CPPUNIT_ASSERT("unkown Exception caught in testeventprocessor2::eventprocessor2Test"==0);
  }
}
