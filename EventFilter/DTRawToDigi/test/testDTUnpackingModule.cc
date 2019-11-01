/**
   \file
   Test suite for DTUnpackingModule

   \author Stefano ARGIRO
   \date 29 Jun 2005

   \note these tests are not testing anything but the thing not crashing
        
*/

#include <cppunit/extensions/HelperMacros.h>
#include <FWCore/Framework/interface/EventProcessor.h>
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
#include <cstdlib>

using namespace std;

string releasetop(std::getenv("SCRAMRT_LOCALRT"));
string testfileLocation = releasetop + "/src/EventFilter/DTRawToDigi/test/";

class testDTUnpackingModule : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDTUnpackingModule);

  CPPUNIT_TEST(testUnpacker);
  CPPUNIT_TEST(testPoolIO);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {
    char* ret = std::getenv("SCRAMRT_LOCALRT");
    if (!ret) {
      cerr << "env variable SCRAMRT_LOCALRT not set, try eval `scramv1 runt -csh`" << endl;
      exit(1);
    }
  }

  void tearDown() {}
  void testUnpacker();
  void writeOut();
  void testPoolIO();
  int runIt(const std::string& config);
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testDTUnpackingModule);

int testDTUnpackingModule::runIt(const std::string& config) {
  edm::AssertHandler ah;
  int rc = 0;
  try {
    edm::EventProcessor proc(config);
    proc.run();
  } catch (seal::Error& e) {
    std::cerr << "Exception caught:  " << e.explainSelf() << std::endl;
    rc = 1;
  }
  return rc;
}

void testDTUnpackingModule::testUnpacker() {
  const std::string config =
      "process TEST = { \n"
      "module dtunpacker = DTUnpackingModule{ }\n"
      "module hit = DummyHitFinderModule{ }\n"
      "path p = {dtunpacker, hit}\n"
      "source = DAQFileInputService{ string fileName =\"" +
      testfileLocation + "dtraw.raw" +
      "\"\n"
      "                              untracked int32 maxEvents = 1 }\n"
      "}\n";

  int rc = runIt(config);
  CPPUNIT_ASSERT(rc == 0);
}

void testDTUnpackingModule::writeOut() {
  const std::string config =
      "process TEST = { \n"
      "module dtunpacker = DTUnpackingModule{ }\n"
      "module out = PoolOutputModule {\n"
      "                   untracked string fileName =\"" +
      testfileLocation + "dtdigis.root" +
      "\"} \n"
      "path p = {dtunpacker, out}\n"
      "source = DAQFileInputService{ string fileName =\"" +
      testfileLocation + "dtraw.raw" +
      "\"\n"
      "                              untracked int32 maxEvents = 1 }\n"
      "}\n";

  int rc = runIt(config);
  CPPUNIT_ASSERT(rc == 0);
}

void testDTUnpackingModule::testPoolIO() {
  writeOut();

  const std::string config =
      "process TEST = { \n"
      " module hit = DummyHitFinderModule{ }\n"
      " path p = {hit}\n"
      " source = PoolSource{ string fileName =\"" +
      testfileLocation + "dtdigis.root" +
      "\"} \n"
      "}\n";

  int rc = runIt(config);
  CPPUNIT_ASSERT(rc == 0);
}
