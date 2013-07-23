/**
   \file
   test file for CSCDetId

   \author Stefano ARGIRO
   \version $Id: testCSCDetId.cc,v 1.5 2007/12/10 18:12:52 ptc Exp $
   \date 27 Jul 2005
*/

static const char CVSId[] = "$Id: testCSCDetId.cc,v 1.5 2007/12/10 18:12:52 ptc Exp $";

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
  CPPUNIT_TEST(testStatic);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp(){}
  void tearDown(){}
  void testOne();
  void testFail();
  void testStatic();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCSCDetId);

void testCSCDetId::testOne(){

  /*
    std::cout << "\ntestCSCDetId: testOne starting... " << std::endl;
  
    std::cout << "min endcap = " << CSCDetId::minEndcapId() << std::endl;
    std::cout << "max endcap = " << CSCDetId::maxEndcapId() << std::endl;
  
    std::cout << "min station = " << CSCDetId::minStationId() << std::endl;
    std::cout << "max station = " << CSCDetId::maxStationId() << std::endl;
  */

  unsigned short rings[]={4,2,2,2};
  unsigned short chambers[]={36,36,36,36, 18,36, 18,36, 18, 36};

  for (int endcap=CSCDetId::minEndcapId(); endcap<=CSCDetId::maxEndcapId(); ++endcap)
    for (int station=CSCDetId::minStationId(); station <= CSCDetId::maxStationId() ; ++station)
      for (int ring=CSCDetId::minRingId(); ring<=rings[station-1]; ++ring) {
        unsigned short itype = CSCDetId::iChamberType( station, ring );
        unsigned short nchambers = chambers[itype-1];
	for (int chamber=CSCDetId::minChamberId(); chamber<=nchambers; ++chamber)
	  for (int layer=CSCDetId::minLayerId(); layer<=CSCDetId::maxLayerId(); ++layer){

	    CSCDetId detid(endcap, station, ring, chamber, layer);
	    //                std::cout << "detid = " << detid.rawId() << "  " << hex << detid.rawId() << 
	    //	    	      "  " << oct << detid.rawId() << dec << std::endl;
	    //	    	    std::cout << "\ndetid.endcap()= " << detid.endcap() << " endcap = " << endcap << std::endl;
	    CPPUNIT_ASSERT(detid.endcap() == endcap);
	    //	    	    std::cout << "\ndetid.station()= " << detid.station() << " station = " << station << std::endl;
            CPPUNIT_ASSERT(detid.station() == station);
            CPPUNIT_ASSERT(detid.ring() == ring);
	    CPPUNIT_ASSERT(detid.chamber() == chamber);
            CPPUNIT_ASSERT(detid.layer() == layer);

	    // check chamber type number	    
	    //            std::cout << "E" << endcap << " S" << station << " R" << ring << " C" << chamber <<
	    //	      " L" << layer << "   chamber type=" << detid.iChamberType() << std::endl;
            CPPUNIT_ASSERT(detid.iChamberType() == CSCDetId::iChamberType( station, ring ) );

	    // test constructor from id
	    int myId = detid.rawId();
	    CSCDetId anotherId(myId);
	    CPPUNIT_ASSERT(detid==anotherId);
	  }
      }
}


void testCSCDetId::testFail(){
  
  // std::cout << "\ntestCSCDetId: testFail starting... " << std::endl;

  // construct using an invalid input index
  try {
    // Invalid layer
    CSCDetId detid(3,1,1,1,7);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0); 
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    //    std::cout << "\ntestCSCDetId: testFail exception caught " << std::endl;
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
    //    std::cout << "\ntestCSCDetId: testFail exception caught " << std::endl;
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }
}

void testCSCDetId::testStatic(){
  int ie = 2;
  int is = 3;
  int ir = 2;
  int ic = 10;
  int il = 3;

  int id1 = CSCDetId::rawIdMaker(2, 3, 2, 10, 3);
  int id2 = CSCDetId::rawIdMaker(ie, is, ir, ic, il);
  int id3 = CSCDetId::rawIdMaker(ie, is, ir, ic, 0 ); // all layers i.e. chamber id

  //  std::cout << "\nE" << ie << " S" << is << " R" << ir << " C" << ic
  //       << " L" << il << " has rawId = " << id2 << " (dec) = "
  //       << hex << id2 << " (hex) " << oct << id2 << " (oct)" << dec << std::endl;
  //
  //  std::cout << "\nE" << ie << " S" << is << " R" << ir << " C" << ic
  //       << " L0" << " has rawId = " << id3 << " = "
  //       << hex << id3 << " (hex) " << oct << id3 << " (oct)" << dec << std::endl;

  CPPUNIT_ASSERT(id1 == id2 );
  CPPUNIT_ASSERT(CSCDetId::endcap(id2)  == ie );
  CPPUNIT_ASSERT(CSCDetId::station(id2) == is );
  CPPUNIT_ASSERT(CSCDetId::ring(id2)    == ir );
  CPPUNIT_ASSERT(CSCDetId::chamber(id2) == ic );
  CPPUNIT_ASSERT(CSCDetId::layer(id2)   == il );
  CPPUNIT_ASSERT(CSCDetId::chamber(id3) == ic );
}
