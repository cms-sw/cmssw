#include <cppunit/extensions/HelperMacros.h>

#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDExpandedView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DD4hep/Detector.h"

#include <string>
#include <memory>

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

using namespace cms;
using namespace std;

class testDDExpandedView : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDExpandedView);
  CPPUNIT_TEST(checkExpandedView);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override;
  void tearDown() override {}
  void checkExpandedView();

private:
  string fileName_;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDExpandedView);

void testDDExpandedView::setUp() {
  fileName_ = edm::FileInPath("DetectorDescription/DDCMS/data/cms-2015-muon-geometry.xml").fullPath();
}

void testDDExpandedView::checkExpandedView() {
  const DDDetector det("DUMMY", fileName_);
  DDCompactView cpview(det);
  DDExpandedView epview(cpview);
}
