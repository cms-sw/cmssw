/**
   \file
   Test suit for DTTopology

   \author R.Bellan
   \version 
   \date 

   \note 
*/

#include <cppunit/extensions/HelperMacros.h>
#include <Geometry/DTGeometry/interface/DTTopology.h>


class testDTTopology: public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE(testDTTopology);

  CPPUNIT_TEST(testDTTopologyMeth);

  CPPUNIT_TEST_SUITE_END();

public:

  void testDTTopologyMeth();

  //  void setUp(){}
  // void tearDown(){}  
}; 

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testDTTopology);


void testDTTopology::testDTTopologyMeth(){
  
  int firstWire1 = 1;
  int nChannels = 48;
  float length = 20.;

  DTTopology topology(firstWire1,nChannels,length);

  CPPUNIT_ASSERT( firstWire1 ==  topology.firstChannel());
  CPPUNIT_ASSERT( nChannels ==  topology.channels());
  CPPUNIT_ASSERT(  3 == topology.channel(LocalPoint(topology.wirePosition(3),0,0)) );
  CPPUNIT_ASSERT(  3 == topology.measurementPosition(LocalPoint(topology.wirePosition(3),0,0)).x() );
  CPPUNIT_ASSERT(  3 == topology.channel(
					 topology.localPosition(
								topology.measurementPosition(
											     LocalPoint(topology.wirePosition(3),0,0) ) ) ) );


}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
