#include <cppunit/extensions/HelperMacros.h>

#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DD4hep/Detector.h"

#include <string>
#include <memory>

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

using namespace cms;
using namespace std;

class testDDFilteredView : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDFilteredView);
  CPPUNIT_TEST(checkFilteredView);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override;
  void tearDown() override {}
  void checkFilteredView();

private:
  string fileName_;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDFilteredView);

void testDDFilteredView::setUp() {
  fileName_ = edm::FileInPath("DetectorDescription/DDCMS/data/cms-2015-muon-geometry.xml").fullPath();
}

void testDDFilteredView::checkFilteredView() {
  unique_ptr<DDDetector> det = make_unique<DDDetector>("DUMMY", fileName_);
  DDFilteredView fview(det.get(), det->description()->worldVolume());
}
