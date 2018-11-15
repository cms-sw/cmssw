/* \file testCSCTriggerMapping.cc
 *
 * \author Lindsey Gray
 * Based on template from S. Argiro & N. Amapane
 */

#include <cppunit/extensions/HelperMacros.h>
#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/Framework/interface/EventProcessor.h>
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include <CondFormats/CSCObjects/interface/CSCTriggerMappingFromFile.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include <iostream>
#include <cstdlib>

std::string treleasetop(getenv("CMSSW_BASE"));
std::string tmappingFilePath= treleasetop + "/src/CondFormats/CSCObjects/test/";

class testCSCTriggerMapping: public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE(testCSCTriggerMapping);

  CPPUNIT_TEST(testRead);

  CPPUNIT_TEST_SUITE_END();

public:

  testCSCTriggerMapping() : myName_( "testCSCTriggerMapping" ), 
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


int testCSCTriggerMapping::runIt(const std::string& config){
  edm::AssertHandler ah;
  int rc=0;
  try {
    edm::EventProcessor proc(edm::getPSetFromConfig(config));
    proc.run();
  } catch (cms::Exception& e){
    std::cerr << "Exception caught:  " 
	      << e.what()
	      << std::endl;
    rc=1;
  }
  return rc;
}


void testCSCTriggerMapping::testRead(){
  std::cout << myName_ << ": --- t e s t C S C T r i g g e r M a p p i n g  ---" << std::endl;
  std::cout << "start " << dashedLine << std::endl;

  std::string mappingFileName = tmappingFilePath + "csc_trigger_test_map.txt";

  CSCTriggerMappingFromFile theMapping( mappingFileName );
 
  //theMapping.setDebugV( true );
  
  // Loop over all possible endcaps, stations, sectors, subsectors and cscids
  // TEST CSCTriggerMapping::chamber(...)

  for ( int endcap = 1;  endcap <= 1; ++endcap )
    for ( int station = 2; station <= 3; ++station ) 
      for( int sector = 5; sector <= 5; ++sector )
	for( int cscid = 1; cscid <= 9; ++cscid )
	  { 	    
	    if(station == 1)
	      for(int subs = 1 ; subs <=2 ; ++subs )
		{
		  std::cout << "\n" << myName_ << ": search for sw id for hw labels, endcap= " << endcap <<
		    ", station=" << station << ", sector=" << sector << 
		    ", subsector=" << subs << ", cscid= " << cscid << std::endl;
		  int id = theMapping.chamber(endcap, station, sector, subs, cscid);
		  
		  std::cout << myName_ << ": found chamber rawId = " << id << std::endl;
	    
		  CSCDetId cid( id );
		  
		  std::cout << myName_ << ": from CSCDetId for this chamber, endcap= " << cid.endcap() <<
		    ", station=" << cid.station() << ", ring=" << cid.ring() << 
		    ", chamber=" << cid.chamber() << std::endl;
		  
		  // Now try direct mapping for specific layers 
		  // TEST CSCTriggerMapping::detId(...) 
		  
		  for ( int layer=1; layer<=6; ++layer ) {
		    std::cout << myName_ << ": map layer with hw labels, endcap= " << endcap <<
		      ", station=" << station << ", sector=" << sector << 
		      ", subsector=" << subs << ", cscid= " << cscid << ", layer=" << layer << std::endl;
		    
		    CSCDetId lid = theMapping.detId( endcap, station, sector, subs, cscid, layer );
		    
		    // And check what we've actually selected...
		    std::cout << myName_ << ": from CSCDetId for this layer, endcap= " << lid.endcap() <<
		      ", station=" << lid.station() << ", ring=" << lid.ring() << 
		      ", chamber=" << lid.chamber() << ", layer=" << lid.layer() << std::endl;         
		  }
		  
		}
	    else
	      {
		std::cout << "\n" << myName_ << ": search for sw id for hw labels, endcap= " << endcap <<
		  ", station=" << station << ", sector=" << sector <<
		  ", subsector=" << 0 << ", cscid= " << cscid << std::endl;
		int id = theMapping.chamber(endcap, station, sector, 0, cscid);

		std::cout << myName_ << ": found chamber rawId = " << id << std::endl;

		CSCDetId cid( id );

		std::cout << myName_ << ": from CSCDetId for this chamber, endcap= " << cid.endcap() <<
		  ", station=" << cid.station() << ", ring=" << cid.ring() <<
		  ", chamber=" << cid.chamber() << std::endl;

		// Now try direct mapping for specific layers
		// TEST CSCTriggerMapping::detId(...)

		for ( int layer=1; layer<=6; ++layer ) {
		  std::cout << myName_ << ": map layer with hw labels, endcap= " << endcap <<
		    ", station=" << station << ", sector=" << sector <<
		    ", subsector=" << 0 << ", cscid= " << cscid << ", layer=" << layer << std::endl;

		  CSCDetId lid = theMapping.detId( endcap, station, sector, 0, cscid, layer );

		  // And check what we've actually selected...
		  std::cout << myName_ << ": from CSCDetId for this layer, endcap= " << lid.endcap() <<
		    ", station=" << lid.station() << ", ring=" << lid.ring() <<
		    ", chamber=" << lid.chamber() << ", layer=" << lid.layer() << std::endl;

		}
	      }
	    std::cout << std::endl;
	    
	  }
	  
	  
  std::cout << dashedLine << " end" << std::endl;
}


///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCSCTriggerMapping);

