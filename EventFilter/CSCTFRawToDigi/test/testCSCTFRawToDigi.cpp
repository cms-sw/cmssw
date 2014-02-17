/* \file testCSCTFRawToDigi.cc
 *
 *  $Date: 2010/09/06 09:29:01 $
 *  $Revision: 1.8 $
 *  \author L. Gray , ripped from testDaqSource
 */

#include <cppunit/extensions/HelperMacros.h>
#include <FWCore/Framework/interface/EventProcessor.h>
#include <FWCore/PluginManager/interface/ProblemTracker.h>
#include <FWCore/Utilities/interface/Exception.h>
#include <iostream>
#include <cstdlib>

using namespace std;

string releasetop(getenv("CMSSW_BASE"));
string testfileLocation= releasetop + "/src/EventFilter/CSCTFRawToDigi/test/";

class testCSCTFRawToDigi: public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE(testCSCTFRawToDigi);

  // Test generating digis from raw data
  CPPUNIT_TEST(testCreateDigis);

  CPPUNIT_TEST_SUITE_END();

public:


  void setUp(){
    char * ret = getenv("CMSSW_BASE");
    if (!ret) {
      cerr<< "env variable SCRAMRT_LOCALRT not set, try eval `scramv1 runt -csh`"<< endl;
      exit(1);
    }
  }

  void tearDown(){}

  void testCreateDigis();

  int  runIt(const std::string& config);

};


int testCSCTFRawToDigi::runIt(const std::string& config){
  edm::AssertHandler ah;
  int rc=0;
  try {
    edm::EventProcessor proc(config);
    proc.run();
  } catch (cms::Exception& e){
    std::cerr << "Exception caught:  "
	      << e.explainSelf()
	      << std::endl;
    rc=1;
  }
  return rc;
}


// Read raw data from a file
void testCSCTFRawToDigi::testCreateDigis(){
  cout << endl << endl << " ---- testCSCTFRawToDigi::testCreateDigis ---- "
       << endl << endl;

  const std::string config= "import FWCore.ParameterSet.Config as cms  \n"
                            "process = cms.Process(\"analyzer\")       \n"
                            "process.load(\"EventFilter.CSCTFRawToDigi.csctfunpacker_cfi\")            \n"
                            "process.load(\"EventFilter.CSCTFRawToDigi.csctfpacker_cfi\")              \n"
                            "process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(128) )\n"
                            "process.load(\"FWCore.MessageLogger.MessageLogger_cfi\")                  \n"
                            "process.MessageLogger.cout.placeholder = cms.untracked.bool(False)        \n"
                            "process.MessageLogger.cout.threshold = cms.untracked.string('INFO')       \n"
                            "process.MessageLogger.debugModules = cms.untracked.vstring('*')           \n"
                            "process.source = cms.Source(\"EmptySource\")                              \n"
                            "process.csctfsinglegen = cms.EDProducer(\"CSCTFSingleGen\")               \n"
                            "process.csctfpacker.lctProducer = cms.InputTag(\"csctfsinglegen:\")\n"
                            "process.csctfpacker.mbProducer   = cms.InputTag(\"null:\")      \n"
                            "process.csctfpacker.trackProducer = cms.InputTag(\"null:\")     \n"
                            "process.csctfunpacker.producer = cms.InputTag(\"csctfpacker\",\"CSCTFRawData\")\n"
                            "process.csctfanalyzer = cms.EDAnalyzer(\"CSCTFAnalyzer\",                 \n"
                            "  mbProducer     = cms.untracked.InputTag(\"csctfunpacker:DT\"),          \n"
                            "  lctProducer    = cms.untracked.InputTag(\"csctfunpacker:\"),            \n"
                            "  trackProducer  = cms.untracked.InputTag(\"csctfunpacker:\"),            \n"
                            "  statusProducer = cms.untracked.InputTag(\"csctfunpacker:\")             \n"
                            ")                                                                         \n"
                            "process.p = cms.Path(process.csctfsinglegen*process.csctfpacker*process.csctfunpacker*process.csctfanalyzer) \n";

  int rc = runIt(config);
  CPPUNIT_ASSERT(rc==0);
}

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCSCTFRawToDigi);
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
