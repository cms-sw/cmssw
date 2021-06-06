#include <cppunit/extensions/HelperMacros.h>

#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DD4hep/Detector.h"

#include <string>
#include <memory>
#include <vector>

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

using namespace cms;

class testDDFilteredViewLevel : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDFilteredViewLevel);
  CPPUNIT_TEST(checkFilteredView);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override;
  void tearDown() override {}
  void checkFilteredView();

private:
  std::string fileName_;
  std::vector<int> refPos_{0, 0, 6, 2, 2};
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDFilteredViewLevel);

void testDDFilteredViewLevel::setUp() {
  fileName_ = edm::FileInPath("Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2021.xml").fullPath();
}

void testDDFilteredViewLevel::checkFilteredView() {
  std::unique_ptr<DDDetector> det = std::make_unique<DDDetector>("DUMMY", fileName_);
  DDFilteredView fview(det.get(), det->description()->worldVolume());
  bool doContinue(true);
  int count = 1;
  auto testPos = fview.navPos();
  while (fview.next(0) && doContinue) {
    std::cout << "#" << count << ": ";
    std::cout << fview.level() << " level: " << fview.name() << " is a "
              << cms::dd::name(cms::DDSolidShapeMap, fview.shape()) << "\n";
    std::cout << "Full path to it is " << fview.path() << "\n";
    auto copyNos = fview.copyNos();
    for (auto it : copyNos) {
      std::cout << it << ", ";
    }
    std::cout << "\n";
    auto pos = fview.navPos();
    for (auto it : pos) {
      std::cout << it << ", ";
    }

    fview.parent();
    std::cout << "\n"
              << fview.level() << " level: " << fview.name() << " is a "
              << cms::dd::name(cms::DDSolidShapeMap, fview.shape()) << "\n";

    if (count == 45) {
      testPos = fview.navPos();
    }
    if (count == 50) {
      doContinue = false;
    }
    count++;
    std::cout << "\n";
  }
  fview.goTo(testPos);
  std::cout << "Go to #45\n";
  std::cout << fview.level() << " level: " << fview.name() << " is a "
            << cms::dd::name(cms::DDSolidShapeMap, fview.shape()) << "\n";
  std::cout << "Full path to it is " << fview.path() << "\n";
  auto copyNos = fview.copyNos();
  for (auto it : copyNos) {
    std::cout << it << ", ";
  }
  std::cout << "\n";
  count = 0;
  auto pos = fview.navPos();
  for (auto it : pos) {
    std::cout << it << ", ";
    CPPUNIT_ASSERT(it == testPos[count++]);
  }
  std::cout << "\n";
  count = 0;
  for (auto it : testPos) {
    std::cout << it << ", ";
    CPPUNIT_ASSERT(it == refPos_[count++]);
  }
  std::cout << "\n";
}
