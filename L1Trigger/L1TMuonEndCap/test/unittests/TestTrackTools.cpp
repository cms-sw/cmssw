#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "L1Trigger/L1TMuonEndCap/interface/TrackTools.h"


class TestTrackTools: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestTrackTools);
  CPPUNIT_TEST(test_pt);
  CPPUNIT_TEST(test_eta);
  CPPUNIT_TEST(test_theta);
  CPPUNIT_TEST(test_phi);
  CPPUNIT_TEST_SUITE_END();

public:
  TestTrackTools() {}
  ~TestTrackTools() {}
  void setUp() {}
  void tearDown() {}

  void test_pt();
  void test_eta();
  void test_theta();
  void test_phi();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestTrackTools);


using namespace emtf;

void TestTrackTools::test_pt()
{
  for (int i = 0; i < 2048; ++i) {
    double pt = 0.5 * static_cast<double>(i);  // pt=[0,1024,step=0.5]
    double eps = 0.5;
    CPPUNIT_ASSERT_DOUBLES_EQUAL(std::min(pt, 255.), calc_pt(calc_pt_GMT(pt)), eps);
  }
}

void TestTrackTools::test_eta()
{
  for (int i = 0; i < 501; ++i) {
    double eta = -2.5 + 0.01 * static_cast<double>(i);  // eta=[-2.5,2.5,step=0.01]
    double eps = 0.010875;
    CPPUNIT_ASSERT_DOUBLES_EQUAL(eta, calc_eta(calc_eta_GMT(eta)), eps);
  }
}

void TestTrackTools::test_theta()
{
  for (int i = 0; i < 1801; ++i) {
    double theta = 0. + 0.1 * static_cast<double>(i);  // theta=[0,180,step=0.1]
    int endcap = (theta >= 90.) ? 2 : 1;
    double eps = 0.28515625;
    CPPUNIT_ASSERT_DOUBLES_EQUAL((endcap == 2 ? (180. - theta) : theta), calc_theta_deg_from_int(calc_theta_int(theta, endcap, 7)), eps);
  }
}

void TestTrackTools::test_phi()
{
  for (int i = 0; i < 3600; ++i) {
    double phi = -180. + 0.1 * static_cast<double>(i);  // phi=[-180,180,step=0.1]
    int sector = (phi - 15. < 0.) ? int((phi - 15. + 360.)/60.) + 1 : int((phi - 15.)/60.) + 1;
    int neigh_sector = (phi - 15. + 22. < 0.) ? int((phi - 15. + 22. + 360.)/60.) + 1 : int((phi - 15. + 22.)/60.) + 1;
    double eps = 360./576;
    CPPUNIT_ASSERT_DOUBLES_EQUAL(phi, calc_phi_GMT_deg(calc_phi_GMT_int(phi)), eps);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(phi, calc_phi_GMT_deg(calc_phi_GMT_int(phi+360.)), eps);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(phi, calc_phi_GMT_deg(calc_phi_GMT_int(phi-360.)), eps);

    eps = 1./60;
    CPPUNIT_ASSERT_DOUBLES_EQUAL(phi, calc_phi_glob_deg(calc_phi_loc_deg(calc_phi_loc_int(phi, sector, 13)), sector), eps);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(phi, calc_phi_glob_deg(calc_phi_loc_deg(calc_phi_loc_int(phi+360., sector, 13)), sector), eps);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(phi, calc_phi_glob_deg(calc_phi_loc_deg(calc_phi_loc_int(phi-360., sector, 13)), sector), eps);
    if ((sector == 6 && neigh_sector == 1) || (sector != 6 && neigh_sector == sector+1)) {
      CPPUNIT_ASSERT_DOUBLES_EQUAL(phi, calc_phi_glob_deg(calc_phi_loc_deg(calc_phi_loc_int(phi, neigh_sector, 13)), neigh_sector), eps);
      CPPUNIT_ASSERT_DOUBLES_EQUAL(phi, calc_phi_glob_deg(calc_phi_loc_deg(calc_phi_loc_int(phi+360., neigh_sector, 13)), neigh_sector), eps);
      CPPUNIT_ASSERT_DOUBLES_EQUAL(phi, calc_phi_glob_deg(calc_phi_loc_deg(calc_phi_loc_int(phi-360., neigh_sector, 13)), neigh_sector), eps);
    }
  }
}
