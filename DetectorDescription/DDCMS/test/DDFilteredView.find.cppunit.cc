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

class testDDFilteredViewFind : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDFilteredViewFind);
  CPPUNIT_TEST(checkFilteredView);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override;
  void tearDown() override {}
  void checkFilteredView();

private:
  void printMe(const cms::DDFilteredView&);
  double refRadLength_ = 0.03142;
  double refXi_ = 6.24526e-05;
  int refCopyNoTag_ = 1000;
  int refCopyNoOffset_ = 100;
  int refCopyNo_ = 1;
  std::string fileName_;
  std::vector<int> refPos_{0, 0, 6, 2, 2};
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDFilteredViewFind);

void testDDFilteredViewFind::setUp() {
  fileName_ = edm::FileInPath("Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2021.xml").fullPath();
}

void testDDFilteredViewFind::checkFilteredView() {
  std::unique_ptr<DDDetector> det = std::make_unique<DDDetector>("DUMMY", fileName_);
  DDFilteredView fview(det.get(), det->description()->worldVolume());

  int count = 1;
  auto testPos = fview.navPos();
  while (fview.next(0)) {
    std::cout << "#" << count << ": ";
    printMe(fview);

    if (count == 45) {
      testPos = fview.navPos();
    }
    if (count == 50) {
      break;
    }
    count++;
  }

  // world_volume/OCMS_1/CMSE_1/Tracker_1/PixelBarrel_1/pixbarlayer0:PixelBarrelLayer0_1/PixelBarrelLadderFull0_6/PixelBarrelModuleBoxFull_1/PixelBarrelModuleFullPlus_4/PixelBarrelSensorFull_1/PixelBarrelActiveFull0_1
  //
  std::vector<int> activeVol{0, 0, 6, 2, 93, 12, 1, 7, 1, 0};
  fview.goTo(activeVol);
  printMe(fview);
  fview.findSpecPar("TrackerRadLength", "TrackerXi");

  double radLength = fview.getNextValue("TrackerRadLength");
  double xi = fview.getNextValue("TrackerXi");
  CPPUNIT_ASSERT(radLength == refRadLength_);
  CPPUNIT_ASSERT(xi == refXi_);

  std::cout << "TrackerRadLength = " << radLength << "\nTrackerXi = " << xi << "\n";

  std::cout << "\n==== Let's go to #45\n";
  fview.goTo(testPos);
  printMe(fview);

  int i = 0;
  for (auto it : fview.navPos()) {
    CPPUNIT_ASSERT(it == testPos[i++]);
  }
  i = 0;
  for (auto it : testPos) {
    CPPUNIT_ASSERT(it == refPos_[i++]);
  }

  // Start with Muon
  std::cout << "\n==== Let's go to Muon\n";
  fview.goTo({0, 0, 8});
  printMe(fview);

  CPPUNIT_ASSERT(fview.name() == "MUON");

  // Go to the first daughter
  fview.next(0);
  printMe(fview);

  // Use it as an escape level
  int startLevel = fview.level();

  count = 1;

  do {
    std::cout << "#" << count++ << ": ";
    std::cout << "started at level " << startLevel << "\n";
    printMe(fview);

  } while (fview.next(0) && fview.level() < startLevel);

  std::cout << "\n==== Continue iteration\n";

  count = 1;
  fview.next(0);
  startLevel = fview.level();
  printMe(fview);

  do {
    std::cout << "#" << count++;
    std::cout << " started at level " << startLevel << ":\n";
    printMe(fview);
  } while (fview.next(0) && fview.level() < startLevel);

  fview.next(0);
  printMe(fview);

  std::cout << "\n==== Let's do it again, go to Muon\n";
  fview.goTo({0, 0, 8});
  printMe(fview);
  CPPUNIT_ASSERT(fview.name() == "MUON");

  // Go to the first daughter
  fview.next(0);
  printMe(fview);

  // Use it as an escape level
  startLevel = fview.level();

  count = 1;

  do {
    std::cout << "#" << count++ << ": ";
    std::cout << "started at level " << startLevel << "\n";
    printMe(fview);

  } while (fview.next(0) && fview.level() < startLevel);

  fview.goTo({0, 0, 8, 0});
  printMe(fview);
  fview.findSpecPar("CopyNoTag", "CopyNoOffset");

  auto tag = fview.getNextValue("CopyNoTag");
  auto offset = fview.getNextValue("CopyNoOffset");
  std::cout << "CopyNoTag = " << tag << "\n";
  std::cout << "CopyNoOffset = " << fview.getNextValue("CopyNoOffset") << "\n";
  CPPUNIT_ASSERT(refCopyNoTag_ == tag);
  CPPUNIT_ASSERT(refCopyNoOffset_ == offset);

  const auto& nodes = fview.history();
  int ctr(0);
  for (const auto& t : nodes.tags) {
    std::cout << t << ": " << nodes.offsets[ctr] << ", " << nodes.copyNos[ctr] << "\n";
    CPPUNIT_ASSERT(refCopyNoTag_ == t);
    CPPUNIT_ASSERT(refCopyNoOffset_ == nodes.offsets[ctr]);
    CPPUNIT_ASSERT(refCopyNo_ == nodes.copyNos[ctr]);
    ctr++;
  }
}

void testDDFilteredViewFind::printMe(const cms::DDFilteredView& fview) {
  std::cout << ">>> " << fview.level() << " level: " << fview.name() << " is a "
            << cms::dd::name(cms::DDSolidShapeMap, fview.shape()) << "\n";
  std::cout << "Full path to it is " << fview.path() << "\n";

  auto copies = fview.copyNos();
  std::cout << "    copy Nos: ";
  std::for_each(copies.rbegin(), copies.rend(), [](const auto& it) { std::cout << it << ", "; });
  std::cout << "\n    levels  : ";
  for (auto it : fview.navPos()) {
    std::cout << it << ", ";
  }
  std::cout << "\n";
}
