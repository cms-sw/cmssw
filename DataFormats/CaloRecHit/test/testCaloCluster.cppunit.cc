/* Unit test for CaloCluster
   Stefano Argiro', Dec 2010

 */

#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "DataFormats/DetId/interface/DetId.h"

class testCaloCluster : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testCaloCluster);
  CPPUNIT_TEST(FlagsTest);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}

  void FlagsTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCaloCluster);

void testCaloCluster::FlagsTest() {
  using namespace reco;
  using namespace math;

  CaloCluster c1(0., XYZPoint(), CaloID(), CaloCluster::island, CaloCluster::cleanOnly);
  CPPUNIT_ASSERT(c1.flags() == CaloCluster::cleanOnly);
  CPPUNIT_ASSERT(c1.isInClean() == true);
  CPPUNIT_ASSERT(c1.isInUnclean() == false);

  CaloCluster c2(0., XYZPoint(), CaloID(), CaloCluster::island, CaloCluster::uncleanOnly);
  CPPUNIT_ASSERT(c2.flags() == CaloCluster::uncleanOnly);
  CPPUNIT_ASSERT(c2.isInUnclean() == true);
  CPPUNIT_ASSERT(c2.isInClean() == false);

  CaloCluster c3(0., XYZPoint(), CaloID(), CaloCluster::island, CaloCluster::common);
  CPPUNIT_ASSERT(c3.flags() == CaloCluster::common);
  CPPUNIT_ASSERT(c3.isInUnclean() == true);
  CPPUNIT_ASSERT(c3.isInClean() == true);

  c3.setFlags(CaloCluster::common);
  CPPUNIT_ASSERT(c3.isInUnclean() == true);
  CPPUNIT_ASSERT(c3.isInClean() == true);
  CPPUNIT_ASSERT(c3.flags() == CaloCluster::common);

  c3.setFlags(CaloCluster::uncleanOnly);
  CPPUNIT_ASSERT(c3.isInUnclean() == true);
  CPPUNIT_ASSERT(c3.isInClean() == false);
  CPPUNIT_ASSERT(c3.flags() == CaloCluster::uncleanOnly);

  c3.setFlags(CaloCluster::cleanOnly);
  CPPUNIT_ASSERT(c3.isInUnclean() == false);
  CPPUNIT_ASSERT(c3.isInClean() == true);
  CPPUNIT_ASSERT(c3.flags() == CaloCluster::cleanOnly);
}
