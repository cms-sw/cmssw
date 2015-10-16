/**
   \file
   test file for ME0DetId

   \author Marcello Maggi from an example of Stefano ARGIRO
   \date 06 Jan 2014
*/

#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
using namespace std;

class testME0DetId: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testME0DetId);
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
CPPUNIT_TEST_SUITE_REGISTRATION(testME0DetId);

void testME0DetId::testOne(){


  for (int region=ME0DetId::minRegionId; region<=ME0DetId::maxRegionId; ++region)
  {
    for (int layer=ME0DetId::minLayerId; layer<=ME0DetId::maxLayerId; ++layer)
      for (int chamber=ME0DetId::minChamberId; chamber<=ME0DetId::maxChamberId; ++chamber){
	for (int roll=ME0DetId::minRollId; roll<=ME0DetId::maxRollId; ++roll){


	  ME0DetId detid(region, layer, chamber, roll);
	  CPPUNIT_ASSERT(detid.region() == region);
	  CPPUNIT_ASSERT(detid.layer() == layer);
	  CPPUNIT_ASSERT(detid.chamber() == chamber);
	  CPPUNIT_ASSERT(detid.roll() == roll);

	         //  test constructor from id
	  int myId = detid.rawId();
	  ME0DetId anotherId(myId);
	  CPPUNIT_ASSERT(detid==anotherId);
	  
	}
      }
  }
}


void testME0DetId::testFail(){


  // contruct using an invalid input index
  try {
    // Wrong Layer
    ME0DetId detid(0,57,0,0);
    CPPUNIT_ASSERT("Failed to throw required exception 0" == 0);      
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception 0" == 0);
  }

  try {
    // Wrong Region
    ME0DetId detid(2,0,0,0);
    CPPUNIT_ASSERT("Failed to throw required exception 1" == 0);      
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception 1" == 0);
  }

  try {
    // Not existant chamber number
    ME0DetId detid(-1,1,37,1);
    CPPUNIT_ASSERT("Failed to throw required exception 2" == 0);      
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception 2" == 0);
  }
  
  // contruct using an invalid input id
  try {
    ME0DetId detid(100);
    CPPUNIT_ASSERT("Failed to throw required exception 3" == 0);      
    detid.rawId(); // avoid compiler warning
  } catch (cms::Exception& e) {
    // OK
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception 3" == 0);
  }

}


void testME0DetId::testMemberOperators(){
  ME0DetId unit1(1,5,3,1);
  ME0DetId unit2=unit1;
  
  CPPUNIT_ASSERT(unit2==unit1);

  ME0DetId layer = unit1.layerId();
  ME0DetId unit3(1,5,3,0);

  CPPUNIT_ASSERT(layer==unit3);

  ME0DetId chamber = unit1.chamberId();
  ME0DetId unit4(1,0,3,0);

  CPPUNIT_ASSERT(chamber==unit4);
  

}
