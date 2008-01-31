/* \file testCSCTFRawToDigi.cc
 *
 *  $Date: 2008/01/22 19:13:14 $
 *  $Revision: 1.5 $
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

  const std::string config=
    "process TEST = { \n"
    "source = DaqSource{ string reader = \"CSCTFFileReader\"\n"
    "                    untracked int32 maxEvents = 10\n"
    "                    PSet pset = { string fileName = \"" + testfileLocation+ "testdata.bin" +"\"}} \n"
    "module csctfunpacker = CSCTFUnpacker{ untracked bool TestBeamData = true\n"
    "                                      untracked int32 TBFedId = 4       \n"
    "                                      untracked int32 TBEndcap = 1      \n"
    "                                      untracked int32 TBSector = 5      \n"
    "                                      untracked string MappingFile = \"src/EventFilter/CSCTFRawToDigi/test/testmapping.map\" }\n"
    "module csctfvalidator = CSCTFValidator { untracked int32 TBFedId = 4 \n"
    "                                         untracked int32 TBEndcap = 1 \n"
    "                                         untracked int32 TBSector = 5 \n"
    "                                         untracked string MappingFile = \"src/EventFilter/CSCTFRawToDigi/test/testmapping.map\" }\n"
    "module poolout = PoolOutputModule {\n"
    "                            untracked string fileName = \"" + testfileLocation + "digis.root" +"\"} \n"
    "path p = {csctfunpacker,csctfvalidator}\n"
    "endpath e = {poolout} \n" 
  "}\n";
  
  int rc = runIt(config);  
  CPPUNIT_ASSERT(rc==0);
}

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCSCTFRawToDigi);
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
