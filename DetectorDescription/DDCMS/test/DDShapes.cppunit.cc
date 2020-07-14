#include <cppunit/extensions/HelperMacros.h>

#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDShapes.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DD4hep/Detector.h"

#include <iostream>

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

using namespace cms;
using namespace cms::dd;
using namespace std;

class testDDShapes : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDShapes);
  CPPUNIT_TEST(checkDDShapes);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override;
  void tearDown() override {}
  void checkDDShapes();

private:
  string fileName_;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDShapes);

void testDDShapes::setUp() {
  fileName_ = edm::FileInPath("DetectorDescription/DDCMS/data/cms-test-shapes.xml").fullPath();
}

void testDDShapes::checkDDShapes() {
  unique_ptr<DDDetector> det = make_unique<DDDetector>("DUMMY", fileName_);
  DDFilteredView fview(det.get(), det->description()->worldVolume());
  std::string prevShape{"null"};
  while (fview.next(0)) {
    std::string title(fview.solid()->GetTitle());
    if (title == prevShape)
      continue;
    prevShape = title;
    cms::DDSolidShape theShape{fview.shape()};
    std::string name(dd::name(DDSolidShapeMap, theShape));
    std::cout << fview.name() << " is a " << title << " == " << name << "\n";
    switch (theShape) {
    case cms::DDSolidShape::ddbox :
    {
      DDBox box(fview);
      if (box.valid) {
        std::cout << "Box (x, y, z) = (" << box.halfX() << ", " << box.halfY() << ", " << box.halfZ() << ")" << std::endl;
      } else {
        std::cout << "Box invalid" << std::endl;
      }
      break;
    }
    case cms::DDSolidShape::ddtubs :
    {
      DDTubs tubs(fview);
      if (tubs.valid) {
        std::cout << "Tube segment (rIn, rOut, zhalf) = (" << tubs.rIn() << ", " << tubs.rOut() << ", " << tubs.zhalf() << ")" << std::endl;
        std::cout << "Tube segment (startPhi, deltaPhi) = (" << tubs.startPhi() << ", " << tubs.deltaPhi() << ")" << std::endl;
      } else {
        std::cout << "Tube invalid" << std::endl;
      }
      break;
    }
    case cms::DDSolidShape::ddtrap :
    {
      DDTrap trap(fview);
      if (trap.valid) {
        std::cout << "Trap (x1, y1) = (" << trap.x1() << ", " << trap.x2() << ")" << std::endl;
        std::cout << "Trap (x2, y2) = (" << trap.x2() << ", " << trap.x2() << ")" << std::endl;
        std::cout << "Trap (x3, x4) = (" << trap.x3() << ", " << trap.x4() << ")" << std::endl;
        std::cout << "Trap (alpha1, alpha2) = (" << trap.alpha1() << ", " << trap.alpha2() << ")" << std::endl;
        std::cout << "Trap (theta, phi, halfz) = (" << trap.theta() << ", " << trap.phi() << ", " << trap.halfZ() << ")" << std::endl;
      } else {
        std::cout << "Trap invalid" << std::endl;
      }
      break;
    }
    case cms::DDSolidShape::ddcons :
    {
      DDCons cons(fview);
      if (cons.valid) {
        std::cout << "Cone segment (phiFrom, deltaPhi, zhalf) = (" << cons.phiFrom() << ", " << cons.deltaPhi() << ", " << cons.zhalf() << ")" << std::endl;
        std::cout << "Cone segment (rInMinusZ, rOutMinusZ) = (" << cons.rInMinusZ() << ", " << cons.rOutMinusZ() << ")" << std::endl;
        std::cout << "Cone segment (rInPlusZ, rOutPlusZ) = (" << cons.rInPlusZ() << ", " << cons.rOutPlusZ() << ")" << std::endl;
      } else {
        std::cout << "Cone invalid" << std::endl;
      }
      break;
    }
    case cms::DDSolidShape::ddpolycone :
    {
      DDPolycone pcone(fview.solid());
      std::cout << "Polycone (startPhi, deltaPhi) = (" << pcone.startPhi() << ", " << pcone.deltaPhi() << ")" << std::endl;
      std::cout << "Polycone z values: ";
      for (auto val : pcone.zVec()) {
        std::cout << val << ", ";
      }
      cout << std::endl;
      std::cout << "Polycone rMinVec values: ";
      for (auto val : pcone.rMinVec()) {
        std::cout << val << ", ";
      }
      cout << std::endl;
      std::cout << "Polycone rMaxVec values: ";
      for (auto val : pcone.rMaxVec()) {
        std::cout << val << ", ";
      }
      cout << std::endl;
      break;
    }
    case cms::DDSolidShape::ddpolyhedra :
    {
      DDPolyhedra pholyd(fview.solid());
      std::cout << "Polyhedra (startPhi, deltaPhi, sides) = (" << pholyd.startPhi() << ", " << pholyd.deltaPhi() << ", " << pholyd.sides() << ")" << std::endl;
      std::cout << "Polyhedra z values: ";
      for (auto val : pholyd.zVec()) {
        std::cout << val << ", ";
      }
      cout << std::endl;
      std::cout << "Polyhedra rMinVec values: ";
      for (auto val : pholyd.rMinVec()) {
        std::cout << val << ", ";
      }
      cout << std::endl;
      std::cout << "Polyhedra rMaxVec values: ";
      for (auto val : pholyd.rMaxVec()) {
        std::cout << val << ", ";
      }
      cout << std::endl;
      std::cout << "Polyhedra rVec values: ";
      for (auto val : pholyd.rVec()) {
        std::cout << val << ", ";
      }
      cout << std::endl;
      break;
    }
    case cms::DDSolidShape::ddtrunctubs :
    {
      DDTruncTubs tubs(fview);
      if (tubs.valid) {
        std::cout << "Tube segment (rIn, rOut, zhalf) = (" << tubs.rIn() << ", " << tubs.rOut() << ", " << tubs.zHalf() << ")" << std::endl;
        std::cout << "Tube segment (startPhi, deltaPhi) = (" << tubs.startPhi() << ", " << tubs.deltaPhi() << ")" << std::endl;
        std::cout << "Tube segment (cutAtStart, cutAtDelta, cutInside) = (" << tubs.cutAtStart() << ", " << tubs.cutAtDelta() << ", " << tubs.cutInside() << ")" << std::endl;
      } else {
        std::cout << "Tube invalid" << std::endl;
      }
      break;
    }
    case cms::DDSolidShape::ddextrudedpolygon :
    {
      DDExtrudedPolygon pgon(fview.solid());
      std::cout << "ExPolygon x values: ";
      for (auto val : pgon.xVec()) {
        std::cout << val << ", ";
      }
      cout << std::endl;
      std::cout << "ExPolygon y values: ";
      for (auto val : pgon.yVec()) {
        std::cout << val << ", ";
      }
      cout << std::endl;
      std::cout << "ExPolygon z values: ";
      for (auto val : pgon.zVec()) {
        std::cout << val << ", ";
      }
      cout << std::endl;
      std::cout << "ExPolygon zx values: ";
      for (auto val : pgon.zxVec()) {
        std::cout << val << ", ";
      }
      cout << std::endl;
      std::cout << "ExPolygon zy values: ";
      for (auto val : pgon.zyVec()) {
        std::cout << val << ", ";
      }
      cout << std::endl;
      std::cout << "ExPolygon zscale values: ";
      for (auto val : pgon.zscaleVec()) {
        std::cout << val << ", ";
      }
      cout << std::endl;
      break;
    }
    default:
      cout << "Unsupported shape" << std::endl;
      break;
    }
  }
}
