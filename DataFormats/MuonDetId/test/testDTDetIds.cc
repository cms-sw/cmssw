/**
   \file
   test file for DTChamberId, DTSuperLayerId, DTLayerId and DTWireId

   \author S. Argiro' & G. Cerminara
   \version $Id: testDTDetIds.cc,v 1.6 2007/06/05 21:19:33 namapane Exp $
   \date 27 Jul 2005
*/

//#define TEST_FORBIDDEN_CTORS 

#include <cppunit/extensions/HelperMacros.h>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <DataFormats/MuonDetId/interface/DTSuperLayerId.h>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>
#include <DataFormats/MuonDetId/interface/DTWireId.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <iostream>
using namespace std;

class testDTDetIds: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testDTDetIds);
CPPUNIT_TEST(testConstructors);
CPPUNIT_TEST(testFail);
CPPUNIT_TEST(testMemberOperators);
CPPUNIT_TEST_SUITE_END();

public:
  void setUp(){}
  void tearDown(){}

  void testConstructors();
  void testFail();
  void testMemberOperators();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testDTDetIds);

// Try every possible constructor
void testDTDetIds::testConstructors(){
  for (int wheel=DTWireId::minWheelId; wheel<=DTWireId::maxWheelId; ++wheel) {
    for (int station=DTWireId::minStationId; 
	 station <= DTWireId::maxStationId ; ++station) {
      for (int sector=DTWireId::minSectorId; 
	   sector<=DTWireId::maxSectorId; ++sector) {
	// Build a DTChamberId
	DTChamberId chamberId(wheel, station, sector);
	CPPUNIT_ASSERT(chamberId.wheel() == wheel);
	CPPUNIT_ASSERT(chamberId.station() == station);
	CPPUNIT_ASSERT(chamberId.sector() == sector);
	
	// Test constructor from id
	int chId = chamberId.rawId();
	DTChamberId newChamberId(chId);
	CPPUNIT_ASSERT(newChamberId == chamberId);

	// Test DTChamberId copy constructor
	DTChamberId copyChamberId(newChamberId);
	CPPUNIT_ASSERT(copyChamberId == newChamberId);

	for (int slayer=DTWireId::minSuperLayerId; 
	     slayer<=DTWireId::maxSuperLayerId; ++slayer) {
	  // Build a DTSuperLayerId
	  DTSuperLayerId slId(wheel, station, sector, slayer);
	  CPPUNIT_ASSERT(slId.wheel() == wheel);
	  CPPUNIT_ASSERT(slId.station() == station);
	  CPPUNIT_ASSERT(slId.sector() == sector);
	  CPPUNIT_ASSERT(slId.superlayer() == slayer);

	  // Test constructor from id
	  int sId = slId.rawId();
	  DTSuperLayerId newSlId(sId);
	  CPPUNIT_ASSERT(newSlId == slId);

	  // Test constructor from chamberId and sl number
	  DTSuperLayerId anotherSLId(chamberId, slayer);
	  CPPUNIT_ASSERT(anotherSLId == slId);

	  // Test DTChamberId copy constructor
	  DTChamberId copyChamberIdFromSl(slId);
	  CPPUNIT_ASSERT(copyChamberIdFromSl == chamberId);

	  // Test DTSuperLayerId constructor from raw SL Id
	  DTChamberId copyChamberIdFromRawSl(sId);
	  CPPUNIT_ASSERT(copyChamberIdFromRawSl == chamberId);

	  // Test DTSuperLayerId copy constructor
	  DTSuperLayerId copySlId(slId);
	  CPPUNIT_ASSERT(slId == copySlId);

	  for (int layer=DTWireId::minLayerId; 
	       layer<=DTWireId::maxLayerId; ++layer) {
	    // Build a DTLayerId
	    DTLayerId layerId(wheel, station, sector, slayer, layer);
	    CPPUNIT_ASSERT(layerId.wheel() == wheel);
	    CPPUNIT_ASSERT(layerId.station() == station);
	    CPPUNIT_ASSERT(layerId.sector() == sector);
	    CPPUNIT_ASSERT(layerId.superlayer() == slayer);
	    CPPUNIT_ASSERT(layerId.layer() == layer);

	    // Test constructor from id
	    int lId = layerId.rawId();
	    DTLayerId newLayerId(lId);
	    CPPUNIT_ASSERT(newLayerId == layerId);
	    
	    // Test constructor from chamberId, sl and layer numbers
	    DTLayerId anotherLayerId(chamberId, slayer, layer);
	    CPPUNIT_ASSERT(anotherLayerId == layerId);

	    // Test constructor from slId and layer number
	    DTLayerId anotherLayerId1(slId, layer);
	    CPPUNIT_ASSERT(anotherLayerId1 == layerId);

	    // Test DTChamberId copy constructor
	    DTChamberId copyChamberIdFromLayer(layerId);
	    CPPUNIT_ASSERT(copyChamberIdFromLayer == chamberId);

	    // Test DTSuperLayerId constructor from raw layer Id
	    DTChamberId copyChamberIdFromRawLayer(lId);
	    CPPUNIT_ASSERT(copyChamberIdFromRawLayer == chamberId);
	    
	    // Test DTSuperLayerId copy constructor
	    DTSuperLayerId copySlIdFromLayer(layerId);
	    CPPUNIT_ASSERT(copySlIdFromLayer == slId);

	    // Test DTSuperLayerId constructor from raw layer Id
	    DTSuperLayerId copySlIdFromRawLayer(lId);
	    CPPUNIT_ASSERT(copySlIdFromRawLayer == slId);

	    // Test DTLayerId copy constructor
	    DTLayerId copyLayerId(layerId);
	    CPPUNIT_ASSERT(copyLayerId == layerId);
	    
	    for (int wire=DTWireId::minWireId; 
		 wire<=DTWireId::maxWireId; ++wire) {
	      
	      // Build a wireId
	      DTWireId wireId(wheel, station, sector, slayer, layer, wire);
	      CPPUNIT_ASSERT(wireId.wheel() == wheel);
	      CPPUNIT_ASSERT(wireId.station() == station);
	      CPPUNIT_ASSERT(wireId.sector() == sector);
	      CPPUNIT_ASSERT(wireId.superlayer() == slayer);
	      CPPUNIT_ASSERT(wireId.layer() == layer);
	      CPPUNIT_ASSERT(wireId.wire() == wire);

	      // Test constructor from id
	      int myId = wireId.rawId();
	      DTWireId newWireId(myId);
	      CPPUNIT_ASSERT(wireId == newWireId);
	      
	      // Test constructor from chamberId, sl, layer and wire numbers
	      DTWireId anotherWireId(chamberId, slayer, layer, wire);
	      CPPUNIT_ASSERT(anotherWireId == wireId);

	      // Test constructor from slId and layer and wire numbers
	      DTWireId anotherWireId1(slId, layer, wire);
	      CPPUNIT_ASSERT(anotherWireId1 == wireId);

	      // Test constructor from layerId and wire number
	      DTWireId anotherWireId2(layerId, wire);
	      CPPUNIT_ASSERT(anotherWireId2 == wireId);

	      // Test DTChamberId copy constructor
	      DTChamberId copyChamberIdFromWire(wireId);
	      CPPUNIT_ASSERT(copyChamberIdFromWire == chamberId);

	      // Test DTChamberId constructor from raw wireId
	      DTChamberId copyChamberIdFromRawWire(myId);
	      CPPUNIT_ASSERT(copyChamberIdFromRawWire == chamberId);

	      // Test DTSuperLayerId copy constructor
	      DTSuperLayerId copySlIdFromWire(wireId);
	      CPPUNIT_ASSERT(copySlIdFromWire == slId);

	      // Test DTSuperLayerId constructor from raw wireId
	      DTSuperLayerId copySlIdFromRawWire(myId);
	      CPPUNIT_ASSERT(copySlIdFromRawWire == slId);
	      
	      // Test DTLayerId copy constructor
	      DTLayerId copyLayerIdFromWire(wireId);
	      CPPUNIT_ASSERT(copyLayerIdFromWire == layerId);

	      // Test DTLayerId constructor from raw wireId
	      DTLayerId copyLayerIdFromRawWire(myId);
	      CPPUNIT_ASSERT(copyLayerIdFromRawWire == layerId);

	      // Test DTWireId copy constructor
	      DTWireId copyWireId(wireId);
	      CPPUNIT_ASSERT(copyWireId == wireId);
	    }
	  }
	}
      }
    }
  }
}

void testDTDetIds::testFail(){
  // Contruct a DTChamberId using an invalid input index
  try {
    // Invalid sector
    DTChamberId detid(0,1,15);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);      
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }
  
  // Contruct a DTChamberId using an invalid input id
  try {
    DTChamberId detid(3211);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);      
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }



  // Contruct a DTSuperLayerId using an invalid input index
  try {
    // Invalid superlayer
    DTSuperLayerId detid(0,1,1,5);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);      
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }
  
  // Contruct a DTSuperLayerId using an invalid input id
  try {
    DTSuperLayerId detid(3211);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);      
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }


   // Contruct a DTLayerId using an invalid input index
  try {
    // Invalid layer
    DTLayerId detid(0,1,1,1,7);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);      
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }
  
  // Contruct a DTLayerId using an invalid input id
  try {
    DTLayerId detid(3211);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);      
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }


  // Contruct a DTWireId using an invalid input index
  try {
    // Invalid wire
    DTWireId wireId(0,1,1,1,1,1000);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);      
    wireId.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }


  // Contruct a DTWireId using an invalid input id
  try {
    DTWireId wireId(3211);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);      
    wireId.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }

}


void testDTDetIds::testMemberOperators(){
  int wheel = 2;
  int station = 3;
  int sector = 8;
  int sl = 1;
  int layer = 4;
  int wire = 15;  

  // Test assignement operators from same type
  DTChamberId chamber1(wheel, station, sector);
  DTChamberId chamber2;
  chamber2 = chamber1;
  CPPUNIT_ASSERT(chamber2==chamber1);

  DTSuperLayerId superlayer1(wheel, station, sector, sl);
  DTSuperLayerId superlayer2;
  superlayer2 = superlayer1;  
  CPPUNIT_ASSERT(superlayer2==superlayer1);

  DTLayerId layer1(wheel, station, sector, sl, layer);
  DTLayerId layer2;
  layer2=layer1;
  CPPUNIT_ASSERT(layer2==layer1);

  DTWireId wire1(wheel, station, sector, sl, layer, wire);
  DTWireId wire2;
  wire2=wire1;
  CPPUNIT_ASSERT(wire2==wire1);  


  // Test getter of base id
  DTChamberId chamber3 = superlayer1.chamberId();
  CPPUNIT_ASSERT(chamber3 == chamber1);
  
  DTChamberId chamber4 = layer1.chamberId();
  CPPUNIT_ASSERT(chamber4 == chamber1);
  
  DTChamberId chamber5 = wire1.chamberId();
  CPPUNIT_ASSERT(chamber5 == chamber1);

  DTSuperLayerId superlayer3 = layer1.superlayerId();
  CPPUNIT_ASSERT(superlayer3 == superlayer1);
  
  DTSuperLayerId superlayer4 = wire1.superlayerId();
  CPPUNIT_ASSERT(superlayer4 == superlayer1);

  DTLayerId layer3 = wire1.layerId();
  CPPUNIT_ASSERT(layer3 == layer1);

  // Test assignement operators from derived objects
  DTChamberId chamber6 = superlayer1;
  CPPUNIT_ASSERT(chamber6 == chamber3);

  DTSuperLayerId superlayer6 = layer1;
  CPPUNIT_ASSERT(superlayer6 == superlayer3);

  DTLayerId layer6 = wire1;
  CPPUNIT_ASSERT(layer6 == layer3);


#ifdef TEST_FORBIDDEN_CTORS
  // Forbidden constructors. None of these should be accepted by the compiler!!!

  // It should not be allowed to create a derived from a base 
  // (it would result in invalid IDs)
  DTSuperLayerId s(chamber1);
  DTLayerId      l(superlayer1);
  DTWireId       w(layer1);  
  
  // It is not currently allowed to build any DT id directly from a Detid
  // (would allow the above ones and may prevent proper slicing)
  DetId d;
  DTChamberId    c(d);
  DTSuperLayerId s1(d);
  DTLayerId      l1(d);
  DTWireId       w1(d);  

  // It is not allowed to copy a derived to a base.
  DTChamberId chamber7 = d;
  DTSuperLayerId superlayer7 = chamber1;
  DTLayerId layer7 = superlayer1;
  DTWireId wire7 = layer1;
#endif


}
