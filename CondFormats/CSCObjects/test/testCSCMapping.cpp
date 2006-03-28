/* \file testCSCMapping.cc
 *
 * \author Tim Cox
 * Based on template from S. Argiro & N. Amapane
 */

#include <cppunit/extensions/HelperMacros.h>
#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/Framework/interface/EventProcessor.h>
#include <FWCore/Utilities/interface/ProblemTracker.h>
#include <CondFormats/CSCObjects/interface/CSCReadoutMappingFromFile.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <iostream>
#include <cstdlib>

std::string releasetop(getenv("CMSSW_BASE"));
std::string mappingFilePath= releasetop + "/src/CondFormats/CSCObjects/test/";

class testCSCMapping: public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE(testCSCMapping);

  CPPUNIT_TEST(testRead);

  CPPUNIT_TEST_SUITE_END();

public:

  testCSCMapping() : myName_( "testCSCMapping" ), 
	 dashedLineWidth(104), dashedLine( std::string(dashedLineWidth, '-') )  {}

  void setUp(){
    char * ret = getenv("CMSSW_BASE");
    if (!ret) {
      std::cerr<< "env variable SCRAMRT_LOCALRT not set, try eval `scramv1 runt -sh`"<< std::endl;
      exit(1);
    }
  }

  void tearDown(){}  

  void testRead();

  int  runIt(const std::string& config);

 private:
  const std::string myName_;
  const int dashedLineWidth;
  std::string dashedLine;
 
}; 


int testCSCMapping::runIt(const std::string& config){
  edm::AssertHandler ah;
  int rc=0;
  try {
    edm::EventProcessor proc(config);
    proc.run();
  } catch (cms::Exception& e){
    std::cerr << "Exception caught:  " 
	      << e.what()
	      << std::endl;
    rc=1;
  }
  return rc;
}


void testCSCMapping::testRead(){
  std::cout << myName_ << ": --- t e s t C S C M a p p i n g  ---" << std::endl;
  std::cout << "start " << dashedLine << std::endl;

  std::string mappingFileName = mappingFilePath + "csc_slice_test_map.txt";

  CSCReadoutMappingFromFile theMapping( mappingFileName );
 
  //theMapping.setDebugV( true );

  // The following labels are irrelevant to hardware in slice test
  int tmb = -1;
  int endcap = -1;
  int station = -1;

  // Loop over all possible crates and dmb slots in slice test
  // TEST CSCReadoutMapping::chamber(...)

  for ( int i=0; i<2; ++i ){
    int vmecrate = i;
    for ( int j = 1; j<11; ++j ) {
      if ( j==6 ) continue;
      int dmb = j;

      std::cout << "\n" << myName_ << ": search for sw id for hw labels, endcap= " << endcap <<
                   ", station=" << station << ", vmecrate=" << vmecrate << 
                   ", dmb=" << dmb << ", tmb= " << tmb << std::endl;
      int id = theMapping.chamber(endcap, station, vmecrate, dmb, tmb);
    
      std::cout << myName_ << ": found chamber rawId = " << id << std::endl;
    
      CSCDetId cid( id );

      std::cout << myName_ << ": from CSCDetId for this chamber, endcap= " << cid.endcap() <<
                              ", station=" << cid.station() << ", ring=" << cid.ring() << 
                              ", chamber=" << cid.chamber() << std::endl;

// Now try direct mapping for specific layers 
// TEST CSCReadoutMapping::detId(...) 

      for ( int layer=1; layer<=6; ++layer ) {
        std::cout << myName_ << ": map layer with hw labels, endcap= " << endcap <<
                                ", station=" << station << ", vmecrate=" << vmecrate << 
                                ", dmb=" << dmb << ", tmb= " << tmb << ", layer=" << layer << std::endl;

        CSCDetId lid = theMapping.detId( endcap, station, vmecrate, dmb, tmb, layer );

        // And check what we've actually selected...
        std::cout << myName_ << ": from CSCDetId for this layer, endcap= " << lid.endcap() <<
                                ", station=" << lid.station() << ", ring=" << lid.ring() << 
                                ", chamber=" << lid.chamber() << ", layer=" << lid.layer() << std::endl;         
      }

      std::cout << std::endl;

    }
  }
  std::cout << dashedLine << " end" << std::endl;
}


///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCSCMapping);

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
