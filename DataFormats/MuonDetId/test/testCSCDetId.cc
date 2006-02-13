/**
   \file
   test file for CSCDetId

   \author Stefano ARGIRO
   \version $Id: testCSCDetId.cc,v 1.3 2005/11/07 13:46:57 ptc Exp $
   \date 27 Jul 2005
*/

static const char CVSId[] = "$Id: testCSCDetId.cc,v 1.3 2005/11/07 13:46:57 ptc Exp $";

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

  //  cout << "\ntestCSCDetId: testOne starting... " << endl;
  //
  //  cout << "min endcap = " << CSCDetId::minEndcapId() << endl;
  //  cout << "max endcap = " << CSCDetId::maxEndcapId() << endl;
  //
  //  cout << "min station = " << CSCDetId::minStationId() << endl;
  //  cout << "max station = " << CSCDetId::maxStationId() << endl;

  for (int endcap=CSCDetId::minEndcapId(); 
       endcap<=CSCDetId::maxEndcapId(); ++endcap)
    for (int station=CSCDetId::minStationId(); 
	 station <= CSCDetId::maxStationId() ; ++station)
      for (int ring=CSCDetId::minRingId(); 
	   ring<=CSCDetId::maxRingId(); ++ring)
	for (int chamber=CSCDetId::minChamberId(); 
	     chamber<=CSCDetId::maxChamberId(); ++chamber)
	  for (int layer=CSCDetId::minLayerId(); 
	       layer<=CSCDetId::maxLayerId(); ++layer){

	    CSCDetId detid(endcap, station, ring, chamber, layer);
	    //            cout << "detid = " << detid.rawId() << "  " << hex << detid.rawId() << 
	    //	      "  " << oct << detid.rawId() << dec << endl;
	    //	    cout << "\ndetid.endcap()= " << detid.endcap() << " endcap = " << endcap << endl;
	    CPPUNIT_ASSERT(detid.endcap() == endcap);
	    //	    cout << "\ndetid.station()= " << detid.station() << " station = " << station << endl;
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
  
  // cout << "\ntestCSCDetId: testFail starting... " << endl;

  // construct using an invalid input index
  try {
    // Invalid layer
    CSCDetId detid(3,1,1,1,7);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0); 
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    //    cout << "\ntestCSCDetId: testFail exception caught " << endl;
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
    //    cout << "\ntestCSCDetId: testFail exception caught " << endl;
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

  //  cout << "\nE" << ie << " S" << is << " R" << ir << " C" << ic
  //       << " L" << il << " has rawId = " << id2 << " (dec) = "
  //       << hex << id2 << " (hex) " << oct << id2 << " (oct)" << dec << endl;
  //
  //  cout << "\nE" << ie << " S" << is << " R" << ir << " C" << ic
  //       << " L0" << " has rawId = " << id3 << " = "
  //       << hex << id3 << " (hex) " << oct << id3 << " (oct)" << dec << endl;

  CPPUNIT_ASSERT(id1 == id2 );
  CPPUNIT_ASSERT(CSCDetId::endcap(id2)  == ie );
  CPPUNIT_ASSERT(CSCDetId::station(id2) == is );
  CPPUNIT_ASSERT(CSCDetId::ring(id2)    == ir );
  CPPUNIT_ASSERT(CSCDetId::chamber(id2) == ic );
  CPPUNIT_ASSERT(CSCDetId::layer(id2)   == il );

  CPPUNIT_ASSERT(CSCDetId::chamber(id3) == ic );
}
