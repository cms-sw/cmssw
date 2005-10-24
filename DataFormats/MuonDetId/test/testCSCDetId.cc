/**
   \file
   test file for CSCDetId

   \author Stefano ARGIRO
   \version $Id: testCSCDetId.cc,v 1.1 2005/08/02 15:46:33 argiro Exp $
   \date 27 Jul 2005
*/

static const char CVSId[] = "$Id: testCSCDetId.cc,v 1.1 2005/08/02 15:46:33 argiro Exp $";

#include <cppunit/extensions/HelperMacros.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <iostream>
using namespace std;

class testCSCDetId: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testCSCDetId);
CPPUNIT_TEST(testOne);
CPPUNIT_TEST(testFail);

CPPUNIT_TEST_SUITE_END();

public:
  void setUp(){}
  void tearDown(){}
  void testOne();
  void testFail();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCSCDetId);

void testCSCDetId::testOne(){

  for (int endcap=CSCDetId::minEndcapId; 
       endcap<=CSCDetId::maxEndcapId   ; ++endcap)
    for (int station=CSCDetId::minStationId; 
	 station <= CSCDetId::maxStationId ; ++station)
      for (int ring=CSCDetId::minRingId; 
	   ring<=CSCDetId::maxRingId; ++ring)
	for (int chamber=CSCDetId::minChamberId; 
	     chamber<=CSCDetId::maxChamberId; ++chamber)
	  for (int layer=CSCDetId::minLayerId; 
	       layer<=CSCDetId::maxLayerId; ++layer){

	    CSCDetId detid(endcap, station, ring, chamber, layer);

	    CPPUNIT_ASSERT(detid.endcap() == endcap);
            CPPUNIT_ASSERT(detid.station() == station);
            CPPUNIT_ASSERT(detid.ring() == ring);
	    CPPUNIT_ASSERT(detid.chamber() == chamber);
            CPPUNIT_ASSERT(detid.layer() == layer);
	    
	    // test constructor from id
	    int myId = detid.rawId();
	    CSCDetId anotherId(myId);
	    CPPUNIT_ASSERT(detid==anotherId);
	  }
}


void testCSCDetId::testFail(){
  
  // contruct using an invalid input index
  try {
    // Invalid layer
    CSCDetId detid(1,1,1,1,7);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0); 
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }
  
  // contruct using an invalid input id
  try {
    CSCDetId detid(3211);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);      
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }
}
