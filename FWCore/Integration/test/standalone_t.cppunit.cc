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
  CPPUNIT_TEST(writeFile);
  CPPUNIT_TEST(readFile);
  CPPUNIT_TEST_SUITE_END();


 public:

  void setUp()
  {
    m_handler = std::auto_ptr<edm::AssertHandler>(new edm::AssertHandler());
    // if (theMessageServicePresence.get() == 0) {
    //   theMessageServicePresence = std::auto_ptr<edm::Presence>(edm::PresenceFactory::get()->makePresence("MessageServicePresence").release());
    // }
  }

  void tearDown(){
    m_handler.reset();
  }

  void writeFile();
  void readFile();

 private:

  std::auto_ptr<edm::AssertHandler> m_handler;
  // static std::auto_ptr<edm::Presence> theMessageServicePresence;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testStandalone);

// std::auto_ptr<edm::Presence> testStandalone::theMessageServicePresence;


void testStandalone::writeFile()
{
  std::string configuration("process p = {\n"
			    "  untracked PSet maxEvents = {untracked int32 input = 5}\n"
			    "  source = EmptySource { }\n"
			    "  module m1 = IntProducer { int32 ivalue = 11 }\n"
			    "  module out = PoolOutputModule {\n"
                            "    untracked string fileName = \"testStandalone.root\"\n"
                            "  }"
			    "  path p1 = { m1 }\n"
                            "  endpath e = { out }\n"
			    "}\n");

  std::vector<std::string> defaultServices;
  defaultServices.push_back(std::string("JobReportService"));
  // defaultServices.push_back(std::string("MessageLogger"));

  edm::EventProcessor proc(configuration, defaultServices);
  proc.beginJob();
  proc.run();
  proc.endJob();
}

void testStandalone::readFile()
{
  std::string configuration("process p = {\n"
                            "  source = PoolSource{ untracked vstring fileNames ={\"file:testStandalone.root\"} }\n"
			    "}\n");

  std::vector<std::string> defaultServices;
  defaultServices.push_back(std::string("JobReportService"));
  // defaultServices.push_back(std::string("MessageLogger"));

  edm::EventProcessor proc(configuration, defaultServices);
  proc.beginJob();
  proc.run();
  proc.endJob();
}
