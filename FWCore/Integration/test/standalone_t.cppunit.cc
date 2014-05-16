/*----------------------------------------------------------------------

Test of a non cmsRun executable

Note that the commented out lines are what is necessary to
setup the MessageLogger in this test. Note that tests like
this will hang after 1000 messages are sent to the MessageLogger
if the MessageLogger is not runnning.

----------------------------------------------------------------------*/  

#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/Framework/interface/EventProcessor.h"
// #include "FWCore/Utilities/interface/Presence.h"
// #include "FWCore/PluginManager/interface/PresenceFactory.h"

#include <cppunit/extensions/HelperMacros.h>

#include <memory>
#include <string>

class testStandalone: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testStandalone);
  CPPUNIT_TEST(writeAndReadFile);
  CPPUNIT_TEST_SUITE_END();


 public:

  void setUp()
  {
    m_handler = std::auto_ptr<edm::AssertHandler>(new edm::AssertHandler());
  }

  void tearDown(){
    m_handler.reset();
  }

  void writeAndReadFile();

 private:

  std::auto_ptr<edm::AssertHandler> m_handler;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testStandalone);



void testStandalone::writeAndReadFile()
{
  {
    std::string configuration("import FWCore.ParameterSet.Config as cms\n"
                              "process = cms.Process('TEST')\n"
                              "process.maxEvents = cms.untracked.PSet(\n"
                              "    input = cms.untracked.int32(5)\n"
                              ")\n"
                              "process.source = cms.Source('EmptySource')\n"
                              "process.JobReportService = cms.Service('JobReportService')\n"
                              "process.InitRootHandlers = cms.Service('InitRootHandlers')\n"
                              "process.m1 = cms.EDProducer('IntProducer',\n"
                              "    ivalue = cms.int32(11)\n"
                              ")\n"
                              "process.out = cms.OutputModule('PoolOutputModule',\n"
                              "    fileName = cms.untracked.string('testStandalone.root')\n"
                              ")\n"
                              "process.p = cms.Path(process.m1)\n"
                              "process.e = cms.EndPath(process.out)\n");

    edm::EventProcessor proc(configuration, true);
    proc.beginJob();
    proc.run();
    proc.endJob();
  }

  {
    std::string configuration("import FWCore.ParameterSet.Config as cms\n"
                              "process = cms.Process('TEST1')\n"
                              "process.source = cms.Source('PoolSource',\n"
                              "    fileNames = cms.untracked.vstring('file:testStandalone.root')\n"
                              ")\n"
                              "process.InitRootHandlers = cms.Service('InitRootHandlers')\n"
                              "process.JobReportService = cms.Service('JobReportService')\n"
                              "process.add_(cms.Service('SiteLocalConfigService'))\n"
                             );

    edm::EventProcessor proc(configuration, true);
    proc.beginJob();
    proc.run();
    proc.endJob();
  }
}
