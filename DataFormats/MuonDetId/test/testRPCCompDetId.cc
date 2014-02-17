/**
   \file
   test file for RPCCompDetId

   \author Stefano ARGIRO
   \version $Id: testRPCCompDetId.cc,v 1.1 2011/11/05 10:39:54 mmaggi Exp $
   \date 27 Jul 2005
*/

#include <cppunit/extensions/HelperMacros.h>
#include <DataFormats/MuonDetId/interface/RPCCompDetId.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <iostream>
using namespace std;

class testRPCCompDetId: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testRPCCompDetId);
CPPUNIT_TEST(testOne);
CPPUNIT_TEST(testFail);
CPPUNIT_TEST(testMemberOperators);
CPPUNIT_TEST_SUITE_END();

public:
  void setUp(){}
  void tearDown(){}

  void testOne();
  void testFail();
  void testMemberOperators();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testRPCCompDetId);

void testRPCCompDetId::testOne(){
  for (int region=RPCCompDetId::minRegionId; region<=RPCCompDetId::maxRegionId; ++region)
  {
     const int minRing ( 0 != region ? RPCCompDetId::minRingForwardId : RPCCompDetId::minRingBarrelId ) ;
     const int maxRing ( 0 != region ? RPCCompDetId::maxRingForwardId : RPCCompDetId::maxRingBarrelId ) ;
     const int minSector ( 0 != region ? RPCCompDetId::minSectorForwardId : RPCCompDetId::minSectorBarrelId ) ;
     const int maxSector ( 0 != region ? RPCCompDetId::maxSectorForwardId : RPCCompDetId::maxSectorBarrelId ) ;
    
      for (int ring=minRing; ring<=maxRing; ++ring) 	    
        for (int station=RPCCompDetId::minStationId; station<=RPCCompDetId::maxStationId; ++station)
          for (int sector=minSector; sector<=maxSector; ++sector)
            for (int layer=RPCCompDetId::minLayerId; layer<=RPCCompDetId::maxLayerId; ++layer)
              for (int subSector=RPCCompDetId::minSubSectorId; subSector<=RPCCompDetId::maxSubSectorId; ++subSector){

		RPCCompDetId detid(region, ring, station, sector, layer, subSector,0);
		
		CPPUNIT_ASSERT(detid.region() == region);
		CPPUNIT_ASSERT(detid.ring() == ring);
		CPPUNIT_ASSERT(detid.station() == station);
		CPPUNIT_ASSERT(detid.sector() == sector);
		CPPUNIT_ASSERT(detid.layer() == layer);
		CPPUNIT_ASSERT(detid.subsector() == subSector);
		
	         //  test constructor from id
		int myId = detid.rawId();
		RPCCompDetId anotherId(myId);
		CPPUNIT_ASSERT(detid==anotherId);
		
	      }
  }
}


void testRPCCompDetId::testFail(){

  // contruct using an invalid input index
  try {
    // Station number too high
    RPCCompDetId detid(0,1,7,2,2,1,1);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);      
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }
  
  // contruct using an invalid input id
  try {
    RPCCompDetId detid(100);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);      
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }
}


void testRPCCompDetId::testMemberOperators(){
  RPCCompDetId unit1(0,-2,1,2,2,1,1);
  RPCCompDetId unit2=unit1;
  CPPUNIT_ASSERT(unit2==unit1);
}
