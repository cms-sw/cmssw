/* Unit test for CaloCluster
   Stefano Argiro', Dec 2010

 */

#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include <iostream>


class testSuperCluster: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testSuperCluster);
  CPPUNIT_TEST(PreshowerPlanesTest);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp(){}
  void tearDown(){}

  void PreshowerPlanesTest();

};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testSuperCluster);

void testSuperCluster::PreshowerPlanesTest(){

  using namespace reco;
  using namespace math;

  SuperCluster c1(0.,XYZPoint());

  c1.setPreshowerPlanesStatus(0);
  c1.setFlags(CaloCluster::cleanOnly);
  c1.setPreshowerPlanesStatus(0);
  
  CPPUNIT_ASSERT(c1.flags() == CaloCluster::cleanOnly);
  CPPUNIT_ASSERT(c1.isInClean()   == true);
  CPPUNIT_ASSERT(c1.isInUnclean() == false);

  CPPUNIT_ASSERT(c1.getPreshowerPlanesStatus() == 0);
  
  c1.setFlags(CaloCluster::uncleanOnly);
  c1.setPreshowerPlanesStatus(1);
  CPPUNIT_ASSERT(c1.getPreshowerPlanesStatus() == 1); 
  CPPUNIT_ASSERT(c1.flags() == CaloCluster::uncleanOnly);
  CPPUNIT_ASSERT(c1.isInClean()   == false);
  CPPUNIT_ASSERT(c1.isInUnclean() == true);


  c1.setPreshowerPlanesStatus(2);
  c1.setFlags(CaloCluster::common);

  CPPUNIT_ASSERT(c1.getPreshowerPlanesStatus() == 2); 
  CPPUNIT_ASSERT(c1.flags() == CaloCluster::common);
  CPPUNIT_ASSERT(c1.isInClean()   == true);
  CPPUNIT_ASSERT(c1.isInUnclean() == true);


  c1.setPreshowerPlanesStatus(3);
  c1.setFlags(CaloCluster::uncleanOnly);

  CPPUNIT_ASSERT(c1.getPreshowerPlanesStatus() == 3); 
  CPPUNIT_ASSERT(c1.flags() == CaloCluster::uncleanOnly);
  CPPUNIT_ASSERT(c1.isInClean()   == false);
  CPPUNIT_ASSERT(c1.isInUnclean() == true);


  

}
