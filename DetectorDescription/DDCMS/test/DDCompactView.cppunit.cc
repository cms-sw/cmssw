#include <cppunit/extensions/HelperMacros.h>

#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DD4hep/Detector.h"

#include <string>
#include <memory>

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

using namespace cms;
using namespace std;

class testDDCompactView : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDCompactView);
  CPPUNIT_TEST(checkCompactView);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override;
  void tearDown() override {}
  void checkCompactView();

private:
  string fileName_;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDCompactView);

void testDDCompactView::setUp() {
  fileName_ = edm::FileInPath("Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2021.xml").fullPath();
}

void testDDCompactView::checkCompactView() {
  const DDDetector det("DUMMY", fileName_);
  DDCompactView cpview(det);
}
