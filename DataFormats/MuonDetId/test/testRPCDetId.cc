/**
   \file
   test file for RPCDetId

   \author Stefano ARGIRO
   \version $Id: testRPCDetId.cc,v 1.5 2011/01/14 03:35:58 sunanda Exp $
   \date 27 Jul 2005
*/

#include <cppunit/extensions/HelperMacros.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <iostream>
using namespace std;

class testRPCDetId: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testRPCDetId);
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
CPPUNIT_TEST_SUITE_REGISTRATION(testRPCDetId);

void testRPCDetId::testOne(){


  for (int region=RPCDetId::minRegionId; region<=RPCDetId::maxRegionId; ++region)
  {
     const int minRing ( 0 != region ? RPCDetId::minRingForwardId : RPCDetId::minRingBarrelId ) ;
     const int maxRing ( 0 != region ? RPCDetId::maxRingForwardId : RPCDetId::maxRingBarrelId ) ;
     const int minSector ( 0 != region ? RPCDetId::minSectorForwardId : RPCDetId::minSectorBarrelId ) ;
     const int maxSector ( 0 != region ? RPCDetId::maxSectorForwardId : RPCDetId::maxSectorBarrelId ) ;
     const int minSubSector ( 0 != region ? RPCDetId::minSubSectorForwardId : RPCDetId::minSubSectorBarrelId ) ;
     const int maxSubSector ( 0 != region ? RPCDetId::maxSubSectorForwardId : RPCDetId::maxSubSectorBarrelId ) ;
    
      for (int ring=minRing; ring<=maxRing; ++ring) 	    
        for (int station=RPCDetId::minStationId; station<=RPCDetId::maxStationId; ++station)
          for (int sector=minSector; sector<=maxSector; ++sector)
            for (int layer=RPCDetId::minLayerId; layer<=RPCDetId::maxLayerId; ++layer)
              for (int subSector=minSubSector; subSector<=maxSubSector; ++subSector)
                for (int roll=RPCDetId::minRollId; roll<=RPCDetId::maxRollId; ++roll){

	          RPCDetId detid(region, ring, station, sector, layer, subSector, roll);

	          CPPUNIT_ASSERT(detid.region() == region);
                  CPPUNIT_ASSERT(detid.ring() == ring);
                  CPPUNIT_ASSERT(detid.station() == station);
	          CPPUNIT_ASSERT(detid.sector() == sector);
                  CPPUNIT_ASSERT(detid.layer() == layer);
	          CPPUNIT_ASSERT(detid.subsector() == subSector);
                  CPPUNIT_ASSERT(detid.roll() == roll);

	         //  test constructor from id
	          int myId = detid.rawId();
	          RPCDetId anotherId(myId);
	          CPPUNIT_ASSERT(detid==anotherId);
  
               }
  }


}


void testRPCDetId::testFail(){


  // contruct using an invalid input index
  try {
    // Station number too high
    RPCDetId detid(0,1,7,2,2,1,1);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);      
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }
  
  // contruct using an invalid input id
  try {
    RPCDetId detid(100);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);      
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }

}


void testRPCDetId::testMemberOperators(){
  RPCDetId unit1(0,-2,1,2,2,1,1);
  RPCDetId unit2=unit1;
  
  CPPUNIT_ASSERT(unit2==unit1);

  RPCDetId chamber = unit1.chamberId();
  RPCDetId unit3(0,-2,1,2,2,1,0);

  CPPUNIT_ASSERT(chamber==unit3);
  

}
