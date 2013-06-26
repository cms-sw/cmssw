/* \file testDaqSource.cc
 *
 *  $Date: 2012/10/10 20:42:00 $
 *  $Revision: 1.8 $
 *  \author S. Argiro, N. Amapane - CERN
 */

#include <cppunit/extensions/HelperMacros.h>
#include <FWCore/Framework/interface/EventProcessor.h>
#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/PluginManager/interface/ProblemTracker.h>
#include <iostream>
#include <cstdlib>

using namespace std;

string releasetop(getenv("CMSSW_BASE"));
string testfileLocation= releasetop + "/src/IORawData/DaqSource/test/";

class testDaqSource: public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE(testDaqSource);

  // Test reading from a raw data file
//   CPPUNIT_TEST(testReadFile);
//   CPPUNIT_TEST(testReadFileWritePool);
//   CPPUNIT_TEST(testReadPool);

  // Test generating random FED blocks
  CPPUNIT_TEST(testGenerate);
  CPPUNIT_TEST(testGenerateWritePool);
  CPPUNIT_TEST(testReadPool);


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

  void testReadFile();
  void testReadFileWritePool();

  void testGenerate();
  void testGenerateWritePool();

  void testReadPool();

  int  runIt(const std::string& config);
 
}; 


int testDaqSource::runIt(const std::string& config){
  edm::AssertHandler ah;
  std::vector<std::string> services;
  services.reserve(2);
  services.push_back(std::string("JobReportService"));
  services.push_back(std::string("InitRootHandlers"));
  int rc=0;
  try {
    edm::EventProcessor proc(config, services);
    proc.beginJob();
    proc.run();
    proc.endJob();
  } catch (cms::Exception& e){
    std::cerr << "Exception caught:  " 
	      << e.explainSelf()
	      << std::endl;
    rc=1;
  }
  return rc;
}


// Read raw data from a file
void testDaqSource::testReadFile(){
  cout << endl << endl << " ---- testDaqSource::testReadFile ---- "
       << endl << endl;

  const std::string config=
    "import FWCore.ParameterSet.Config as cms\n"
    "process = cms.Process('TESTReadFile')\n"
    "process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))\n"
    "process.source = cms.Source('DaqSource', readerPluginName = cms.untracked.string('DaqFileReader'), readerPset = cms.untracked.PSet(fileName = cms.string('" + testfileLocation + "rawdt.raw')))\n"
    "process.dummyunpacker = cms.EDAnalyzer('DummyUnpackingModule', fedRawDataCollectionTag = cms.InputTag('rawDataCollector'))\n"
    "process.p = cms.Path(process.dummyunpacker)\n"
    "\n";
  
  int rc = runIt(config);
  CPPUNIT_ASSERT(rc==0);
}


// Read raw data from a file and write out a pool DB
void testDaqSource::testReadFileWritePool(){
  cout << endl << endl << " ---- testDaqSource::testReadFileWritePool ---- "
       << endl << endl;

  const std::string config=
    "import FWCore.ParameterSet.Config as cms\n"
    "process = cms.Process('TESTReadFileWritePool')\n"
    "process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))\n"
    "process.source = cms.Source('DaqSource', readerPluginName = cms.untracked.string('DaqFileReader'), readerPset = cms.untracked.PSet(fileName = cms.string('" + testfileLocation + "rawdt.raw')))\n"
    "process.dummyunpacker = cms.EDAnalyzer('DummyUnpackingModule', fedRawDataCollectionTag = cms.InputTag('rawDataCollector'))\n"
    "process.out = cms.OutputModule('PoolOutputModule', fileName = cms.untracked.string('" + testfileLocation + "rawdata.root'))\n"
    "process.p = cms.Path(process.dummyunpacker)\n"
    "process.ep = cms.EndPath(process.out)\n"
    "\n";
 int rc = runIt(config);
 CPPUNIT_ASSERT(rc==0);

}


// Re-read the pool DB
void testDaqSource::testReadPool(){
  cout << endl << endl << " ---- testDaqSource::testReadPool ---- "
       << endl << endl;

  const std::string config=
    "import FWCore.ParameterSet.Config as cms\n"
    "process = cms.Process('TESTReadPool')\n"
    "process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))\n"
    "process.dummyunpacker = cms.EDAnalyzer('DummyUnpackingModule', fedRawDataCollectionTag = cms.InputTag('rawDataCollector'))\n"
    "process.p = cms.Path(process.dummyunpacker)\n"
    "process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring('file:" + testfileLocation + "rawdata.root'))\n"
    "\n";

  int rc = runIt(config);
  CPPUNIT_ASSERT(rc==0);

  
}


// Read raw data from a file
void testDaqSource::testGenerate(){
  cout << endl << endl << " ---- testDaqSource::testGenerate ---- " 
       << endl << endl;

  const std::string config=
    "import FWCore.ParameterSet.Config as cms\n"
    "process = cms.Process('TESTGenerate')\n"
    "process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))\n"
    "process.source = cms.Source('DaqSource', readerPluginName = cms.untracked.string('DaqFakeReader'), readerPset = cms.untracked.PSet( ))\n"
    "process.dummyunpacker = cms.EDAnalyzer('DummyUnpackingModule', fedRawDataCollectionTag = cms.InputTag('rawDataCollector'))\n"
    "process.p = cms.Path(process.dummyunpacker)\n"
    "\n";
  
  int rc = runIt(config);
  CPPUNIT_ASSERT(rc==0);
}


// Read raw data from a file and write out a pool DB
void testDaqSource::testGenerateWritePool(){
  cout << endl << endl << " ---- testDaqSource::testGenerateWritePool ---- " 
       << endl << endl;

  const std::string config=
    "import FWCore.ParameterSet.Config as cms\n"
    "process = cms.Process('TESTGenerateWritePool')\n"
    "process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))\n"
    "process.source = cms.Source('DaqSource', readerPluginName = cms.untracked.string('DaqFakeReader'), readerPset = cms.untracked.PSet( ))\n"
    "process.dummyunpacker = cms.EDAnalyzer('DummyUnpackingModule', fedRawDataCollectionTag = cms.InputTag('rawDataCollector'))\n"
    "process.out = cms.OutputModule('PoolOutputModule', fileName = cms.untracked.string('" + testfileLocation + "rawdata.root'))\n"
    "process.p = cms.Path(process.dummyunpacker)\n"
    "process.ep = cms.EndPath(process.out)\n"
    "\n";
 int rc = runIt(config);
 CPPUNIT_ASSERT(rc==0);

}

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testDaqSource);

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
