/* Unit test for CaloCluster
   Stefano Argiro', Dec 2010

 */

#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include <iostream>

class testSuperCluster : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSuperCluster);
  CPPUNIT_TEST(PreshowerPlanesTest);
  CPPUNIT_TEST(CopyCtorTest);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}

  void PreshowerPlanesTest();
  void CopyCtorTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testSuperCluster);

void testSuperCluster::PreshowerPlanesTest() {
  using namespace reco;
  using namespace math;

  SuperCluster c1(0., XYZPoint());

  c1.setPreshowerPlanesStatus(0);
  c1.setFlags(CaloCluster::cleanOnly);
  c1.setPreshowerPlanesStatus(0);

  CPPUNIT_ASSERT(c1.flags() == CaloCluster::cleanOnly);
  CPPUNIT_ASSERT(c1.isInClean() == true);
  CPPUNIT_ASSERT(c1.isInUnclean() == false);

  CPPUNIT_ASSERT(c1.getPreshowerPlanesStatus() == 0);

  c1.setFlags(CaloCluster::uncleanOnly);
  c1.setPreshowerPlanesStatus(1);
  CPPUNIT_ASSERT(c1.getPreshowerPlanesStatus() == 1);
  CPPUNIT_ASSERT(c1.flags() == CaloCluster::uncleanOnly);
  CPPUNIT_ASSERT(c1.isInClean() == false);
  CPPUNIT_ASSERT(c1.isInUnclean() == true);

  c1.setPreshowerPlanesStatus(2);
  c1.setFlags(CaloCluster::common);

  CPPUNIT_ASSERT(c1.getPreshowerPlanesStatus() == 2);
  CPPUNIT_ASSERT(c1.flags() == CaloCluster::common);
  CPPUNIT_ASSERT(c1.isInClean() == true);
  CPPUNIT_ASSERT(c1.isInUnclean() == true);

  c1.setPreshowerPlanesStatus(3);
  c1.setFlags(CaloCluster::uncleanOnly);

  CPPUNIT_ASSERT(c1.getPreshowerPlanesStatus() == 3);
  CPPUNIT_ASSERT(c1.flags() == CaloCluster::uncleanOnly);
  CPPUNIT_ASSERT(c1.isInClean() == false);
  CPPUNIT_ASSERT(c1.isInUnclean() == true);
}

void testSuperCluster::CopyCtorTest() {
  using namespace reco;
  using namespace edm;

  CaloID id;
  CaloCluster c1(1.0, math::XYZPoint(0, 0, 0), id);
  CaloCluster c2(2.0, math::XYZPoint(0, 0, 0), id);
  CaloCluster c3(3.0, math::XYZPoint(0, 0, 0), id);

  CaloClusterCollection clusters;
  clusters.push_back(c1);
  clusters.push_back(c2);
  clusters.push_back(c3);

  ProductID const pid(1, 1);

  OrphanHandle<CaloClusterCollection> handle(&clusters, pid);

  CaloClusterPtr pc1(handle, 0), pc2(handle, 1), pc3(handle, 2);

  SuperCluster sc(5.0, math::XYZPoint(0, 0, 0));
  sc.setSeed(pc1);
  sc.addCluster(pc2);
  sc.addCluster(pc3);

  SuperCluster sccopy = sc;

  CPPUNIT_ASSERT(sc.energy() == sccopy.energy());

  CPPUNIT_ASSERT(sc.seed()->energy() == sccopy.seed()->energy());

  CaloClusterPtrVector::const_iterator bcitcopy = sccopy.clustersBegin();

  for (CaloClusterPtrVector::const_iterator bcit = sc.clustersBegin(); bcit != sc.clustersEnd(); ++bcit) {
    CPPUNIT_ASSERT((*bcit)->energy() == (*bcitcopy)->energy());

    bcitcopy++;
  }
}
