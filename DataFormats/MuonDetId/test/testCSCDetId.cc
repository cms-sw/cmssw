/**
   \file
   test file for CSCDetId

   \author Stefano ARGIRO
   \version $Id$
   \date 27 Jul 2005
*/

static const char CVSId[] = "$Id$";


#include <iostream>

#include <cppunit/extensions/HelperMacros.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

class testCSCDetId: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testCSCDetId);
CPPUNIT_TEST(testOne);

CPPUNIT_TEST_SUITE_END();

public:
  void setUp(){}
  void tearDown(){}
  void testOne();

};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCSCDetId);

void testCSCDetId::testOne(){

  for (unsigned int endcap=CSCDetId::minEndcapId; 
       endcap<=CSCDetId::maxEndcapId   ; ++endcap)
    for (unsigned int station=CSCDetId::minStationId; 
	 station <= CSCDetId::maxStationId ; ++station)
      for (unsigned int ring=CSCDetId::minRingId; 
	   ring<=CSCDetId::maxRingId; ++ring)
	for (unsigned int chamber=CSCDetId::minChamberId; 
	     chamber<=CSCDetId::maxChamberId; ++chamber)
	  for (unsigned int layer=CSCDetId::minLayerId; 
	       layer<=CSCDetId::maxLayerId; ++layer){

	    CSCDetId detid(endcap, station, ring, chamber, layer);

	    CPPUNIT_ASSERT(detid.endcap() == endcap);
            CPPUNIT_ASSERT(detid.station() == station);
            CPPUNIT_ASSERT(detid.ring() == ring);
	    CPPUNIT_ASSERT(detid.chamber() == chamber);
            CPPUNIT_ASSERT(detid.layer() == layer);
	  }



}
