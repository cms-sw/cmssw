#include <cppunit/extensions/HelperMacros.h>

#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DD4hep/Detector.h"

#include <iostream>

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

using namespace cms;
using namespace std;

class testDDSolid : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDSolid);
  CPPUNIT_TEST(checkDDSolid);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override;
  void tearDown() override {}
  void checkDDSolid();

private:
  string fileName_;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDSolid);

void testDDSolid::setUp() {
  fileName_ = edm::FileInPath("DetectorDescription/DDCMS/data/cms-test-shapes.xml").fullPath();
}

void testDDSolid::checkDDSolid() {
  unique_ptr<DDDetector> det = make_unique<DDDetector>("DUMMY", fileName_);
  DDFilteredView fview(det.get(), det->description()->worldVolume());
  int counter = 0;
  while (fview.next(0)) {
    std::cout << fview.path() << "\n";
    std::string title(fview.solid()->GetTitle());
    std::string name(cms::dd::name(cms::DDSolidShapeMap, fview.shape()));
    std::cout << "#" << counter++ << ": " << fview.name() << "[" << fview.copyNum() << "]"
              << " is a " << title << " == " << name << "\n";
    CPPUNIT_ASSERT(title.compare(name) == 0);

    if ((dd4hep::isA<dd4hep::SubtractionSolid>(fview.solid()) and
         not dd4hep::isA<dd4hep::TruncatedTube>(fview.solid()) and
         not dd4hep::isA<dd4hep::PseudoTrap>(fview.solid()))) {
      DDSolid solid(fview.solid());
      auto solidA = solid.solidA();
      std::cout << "Solid A is a " << solidA->GetTitle() << "\n";
      if (dd4hep::isA<dd4hep::ConeSegment>(dd4hep::Solid(solidA))) {
        cout << " is a ConeSegment:\n";
        for (auto const& i : solidA.dimensions())
          cout << i / dd4hep::cm << " cm, ";
      }
      cout << "\n";
      DDSolid a(solidA);
      for (auto const& i : a.parameters())
        cout << i / dd4hep::cm << " cm, ";
      cout << "\n";

      auto solidB = solid.solidB();
      std::cout << "Solid B is a " << solidB->GetTitle() << "\n";
      if (dd4hep::isA<dd4hep::ConeSegment>(dd4hep::Solid(solidB))) {
        cout << " is a ConeSegment:\n";
        for (auto const& i : solidB.dimensions())
          cout << i / dd4hep::cm << " cm, ";
      }
      cout << "\n";
      DDSolid b(solidB);
      for (auto const& i : b.parameters())
        cout << i / dd4hep::cm << " cm, ";
      cout << "\n";
    }
  }
}
