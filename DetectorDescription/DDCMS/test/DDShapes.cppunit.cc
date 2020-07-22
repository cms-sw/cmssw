#include <cppunit/extensions/HelperMacros.h>

#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDShapes.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/Math/interface/angle_units.h"
#include "DD4hep/Detector.h"

#include <iostream>

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

using namespace cms;
using namespace cms::dd;
using namespace std;
using namespace angle_units::operators;

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

static inline constexpr double convertCmToM(double length) { return (length / 100.); }

void testDDShapes::setUp() {
  fileName_ = edm::FileInPath("DetectorDescription/DDCMS/data/cms-test-shapes.xml").fullPath();
}

void testDDShapes::checkDDShapes() {
  unique_ptr<DDDetector> det = make_unique<DDDetector>("DUMMY", fileName_);
  DDFilteredView fview(det.get(), det->description()->worldVolume());
  std::string prevShape{"null"};
  while (fview.next(0)) {
    std::string title(fview.solid()->GetTitle());
    if (fview.name() == prevShape)
      continue;
    prevShape = fview.name();
    cms::DDSolidShape theShape{fview.shape()};
    std::string name(dd::name(DDSolidShapeMap, theShape));
    std::cout << fview.name() << " is a " << title << " == " << name << "\n";
    switch (theShape) {
      case cms::DDSolidShape::ddbox: {
        DDBox box(fview);
        if (box.valid) {
          std::cout << "Box (x, y, z) = (" << box.halfX() << ", " << box.halfY() << ", " << box.halfZ() << ")"
                    << std::endl;
          if (fview.name() == "box1") {
            if ((box.halfX() != 9.) || (box.halfY() != 10.) || (box.halfZ() != 11.)) {
              std::cout << "ERROR: box1 sides should be dx=9*cm dy=10*cm dz=11*cm" << std::endl;
              CPPUNIT_ASSERT(false);
            } else {
              std::cout << "box1 matches reference." << std::endl;
            }
          }
        } else {
          std::cout << "Box invalid" << std::endl;
        }
        break;
      }
      case cms::DDSolidShape::ddtubs: {
        DDTubs tubs(fview);
        if (tubs.valid) {
          std::cout << "Tube segment (rIn, rOut, zhalf) = (" << tubs.rIn() << ", " << tubs.rOut() << ", "
                    << tubs.zhalf() << ")" << std::endl;
          std::cout << "Tube segment (startPhi, deltaPhi) = (" << tubs.startPhi() << ", " << tubs.deltaPhi() << ")"
                    << std::endl;
          if (fview.name() == "ZStop_2_ZStopSpaceDivision") {
            const std::string referenceStr{"rMin=4.694*m rMax=4.955*m dz=32*cm startPhi=10.75*deg deltaPhi=3.0*deg"};
            char tubeParams[200];
            (void)sprintf(tubeParams,
                          "rMin=%.4g*m rMax=%.4g*m dz=%g*cm startPhi=%.4g*deg deltaPhi=%.1f*deg",
                          convertCmToM(tubs.rIn()),
                          convertCmToM(tubs.rOut()),
                          tubs.zhalf(),
                          convertRadToDeg(tubs.startPhi()),
                          convertRadToDeg(tubs.deltaPhi()));
            if (referenceStr == tubeParams) {
              std::cout << "Correct match with reference: " << referenceStr << std::endl;
            } else {
              std::cout << "ERROR: Mismatch of params for tube segment." << std::endl;
              std::cout << tubeParams << " does not match reference " << std::endl;
              std::cout << referenceStr << std::endl;
              CPPUNIT_ASSERT(referenceStr == tubeParams);
            }
          }
        } else {
          std::cout << "Tube invalid" << std::endl;
        }
        break;
      }
      case cms::DDSolidShape::ddtrap: {
        DDTrap trap(fview);
        if (trap.valid) {
          std::cout << "Trap (x1, y1) = (" << trap.x1() << ", " << trap.y1() << ")" << std::endl;
          std::cout << "Trap (x2, y2) = (" << trap.x2() << ", " << trap.y2() << ")" << std::endl;
          std::cout << "Trap (x3, x4) = (" << trap.x3() << ", " << trap.x4() << ")" << std::endl;
          std::cout << "Trap (alpha1, alpha2) = (" << trap.alpha1() << ", " << trap.alpha2() << ")" << std::endl;
          std::cout << "Trap (theta, phi, halfz) = (" << trap.theta() << ", " << trap.phi() << ", " << trap.halfZ()
                    << ")" << std::endl;
          if (fview.name() == "trap1") {
            const std::string referenceStr{
                "alp1=10*deg alp2=10*deg bl1=30*cm bl2=10*cm dz=60*cm h1=40*cm h2=16*cm phi=5*deg theta=20*deg "
                "tl1=41*cm tl2=14*cm"};
            char shapeParams[400];
            (void)sprintf(shapeParams,
                          "alp1=%g*deg alp2=%g*deg bl1=%g*cm bl2=%g*cm dz=%g*cm h1=%g*cm h2=%g*cm phi=%g*deg "
                          "theta=%g*deg tl1=%g*cm tl2=%g*cm",
                          convertRadToDeg(trap.alpha1()),
                          convertRadToDeg(trap.alpha2()),
                          trap.x1(),
                          trap.x3(),
                          trap.halfZ(),
                          trap.y1(),
                          trap.y2(),
                          convertRadToDeg(trap.phi()),
                          convertRadToDeg(trap.theta()),
                          trap.x2(),
                          trap.x4());
            if (referenceStr == shapeParams) {
              std::cout << "Correct match with reference: " << referenceStr << std::endl;
            } else {
              std::cout << "ERROR: Mismatch of params for trapezoid." << std::endl;
              std::cout << shapeParams << " does not match reference " << std::endl;
              std::cout << referenceStr << std::endl;
              CPPUNIT_ASSERT(referenceStr == shapeParams);
            }
          }
        } else {
          std::cout << "Trap invalid" << std::endl;
        }
        break;
      }
      case cms::DDSolidShape::ddcons: {
        DDCons cons(fview);
        if (cons.valid) {
          std::cout << "Cone segment (phiFrom, deltaPhi, zhalf) = (" << cons.phiFrom() << ", " << cons.deltaPhi()
                    << ", " << cons.zhalf() << ")" << std::endl;
          std::cout << "Cone segment (rInMinusZ, rOutMinusZ) = (" << cons.rInMinusZ() << ", " << cons.rOutMinusZ()
                    << ")" << std::endl;
          std::cout << "Cone segment (rInPlusZ, rOutPlusZ) = (" << cons.rInPlusZ() << ", " << cons.rOutPlusZ() << ")"
                    << std::endl;
          if (fview.name() == "cone1") {
            const std::string referenceStr{
                "dz=1*m rMin1=50*cm rMax1=101*cm rMin2=75*cm rMax2=125*cm startPhi=1*deg deltaPhi=360*deg"};
            char shapeParams[400];
            (void)sprintf(shapeParams,
                          "dz=%g*m rMin1=%g*cm rMax1=%g*cm rMin2=%g*cm rMax2=%g*cm startPhi=%g*deg deltaPhi=%g*deg",
                          convertCmToM(cons.zhalf()),
                          cons.rInMinusZ(),
                          cons.rOutMinusZ(),
                          cons.rInPlusZ(),
                          cons.rOutPlusZ(),
                          convertRadToDeg(cons.phiFrom()),
                          convertRadToDeg(cons.deltaPhi()));
            if (referenceStr == shapeParams) {
              std::cout << "Correct match with reference: " << referenceStr << std::endl;
            } else {
              std::cout << "ERROR: Mismatch of params for cone segment." << std::endl;
              std::cout << shapeParams << " does not match reference " << std::endl;
              std::cout << referenceStr << std::endl;
              CPPUNIT_ASSERT(referenceStr == shapeParams);
            }
          }
        } else {
          std::cout << "Cone invalid" << std::endl;
        }
        break;
      }
      case cms::DDSolidShape::ddpolycone: {
        DDPolycone pcone(fview.solid());
        std::cout << "Polycone (startPhi, deltaPhi) = (" << pcone.startPhi() << ", " << pcone.deltaPhi() << ")"
                  << std::endl;
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
        if (fview.name() == "pczsect2") {
          std::string referenceStr{"startPhi=2*deg deltaPhi=360*deg"};
          referenceStr += "\n<ZSection z=1*m rMin=49*cm rMax=55*cm/>";
          referenceStr += "\n<ZSection z=1.5*m rMin=50*cm rMax=60*cm/>";
          referenceStr += "\n<ZSection z=2*m rMin=51*cm rMax=75*cm/>\n";
          char shapeParams[400];
          (void)sprintf(shapeParams,
                        "startPhi=%g*deg deltaPhi=%g*deg",
                        convertRadToDeg(pcone.startPhi()),
                        convertRadToDeg(pcone.deltaPhi()));
          char fullLine[3][200];
          for (int index = 0; index < 3; ++index) {
            (void)sprintf(fullLine[index], "\n<ZSection z=%g*m ", convertCmToM(pcone.zVec()[index]));
          }
          for (int index = 0; index < 3; ++index) {
            char line[100];
            (void)sprintf(line, "rMin=%g*cm ", pcone.rMinVec()[index]);
            (void)strcat(fullLine[index], line);
          }
          for (int index = 0; index < 3; ++index) {
            char line[100];
            (void)sprintf(line, "rMax=%g*cm/>", pcone.rMaxVec()[index]);
            (void)strcat(fullLine[index], line);
          }
          (void)strcat(shapeParams, fullLine[0]);
          (void)strcat(shapeParams, fullLine[1]);
          (void)strcat(shapeParams, fullLine[2]);
          (void)strcat(shapeParams, "\n");
          if (referenceStr == shapeParams) {
            std::cout << "Correct match with reference: " << referenceStr << std::endl;
          } else {
            std::cout << "ERROR: Mismatch of params for polycone." << std::endl;
            std::cout << shapeParams << " does not match reference " << std::endl;
            std::cout << referenceStr << std::endl;
            CPPUNIT_ASSERT(referenceStr == shapeParams);
          }
        }
        break;
      }
      case cms::DDSolidShape::ddpolyhedra: {
        DDPolyhedra pholyd(fview.solid());
        std::cout << "Polyhedra (startPhi, deltaPhi, sides) = (" << pholyd.startPhi() << ", " << pholyd.deltaPhi()
                  << ", " << pholyd.sides() << ")" << std::endl;
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
        if (fview.name() == "phzsect") {
          std::string referenceStr{"numSide=18 startPhi=2*deg deltaPhi=360*deg"};
          referenceStr += "\n<ZSection z=1*m rMin=50*cm rMax=50*cm/>";
          referenceStr += "\n<ZSection z=1.5*m rMin=49*cm rMax=60*cm/>";
          referenceStr += "\n<ZSection z=2*m rMin=51*cm rMax=75*cm/>\n";
          char shapeParams[400];
          (void)sprintf(shapeParams,
                        "numSide=%d startPhi=%g*deg deltaPhi=%g*deg",
                        pholyd.sides(),
                        convertRadToDeg(pholyd.startPhi()),
                        convertRadToDeg(pholyd.deltaPhi()));
          char fullLine[3][200];
          for (int index = 0; index < 3; ++index) {
            (void)sprintf(fullLine[index], "\n<ZSection z=%g*m ", convertCmToM(pholyd.zVec()[index]));
          }
          for (int index = 0; index < 3; ++index) {
            char line[100];
            (void)sprintf(line, "rMin=%g*cm ", pholyd.rMinVec()[index]);
            (void)strcat(fullLine[index], line);
          }
          for (int index = 0; index < 3; ++index) {
            char line[100];
            (void)sprintf(line, "rMax=%g*cm/>", pholyd.rMaxVec()[index]);
            (void)strcat(fullLine[index], line);
          }
          (void)strcat(shapeParams, fullLine[0]);
          (void)strcat(shapeParams, fullLine[1]);
          (void)strcat(shapeParams, fullLine[2]);
          (void)strcat(shapeParams, "\n");
          if (referenceStr == shapeParams) {
            std::cout << "Correct match with reference: " << referenceStr << std::endl;
          } else {
            std::cout << "ERROR: Mismatch of params for polyhedra." << std::endl;
            std::cout << shapeParams << " does not match reference " << std::endl;
            std::cout << referenceStr << std::endl;
            CPPUNIT_ASSERT(referenceStr == shapeParams);
          }
        }
        break;
      }
      case cms::DDSolidShape::ddtrunctubs: {
        DDTruncTubs tubs(fview);
        if (tubs.valid) {
          std::cout << "Tube segment (rIn, rOut, zhalf) = (" << tubs.rIn() << ", " << tubs.rOut() << ", "
                    << tubs.zHalf() << ")" << std::endl;
          std::cout << "Tube segment (startPhi, deltaPhi) = (" << tubs.startPhi() << ", " << tubs.deltaPhi() << ")"
                    << std::endl;
          std::cout << "Tube segment (cutAtStart, cutAtDelta, cutInside) = (" << tubs.cutAtStart() << ", "
                    << tubs.cutAtDelta() << ", " << tubs.cutInside() << ")" << std::endl;
        } else {
          std::cout << "Tube invalid" << std::endl;
        }
        if (fview.name() == "trunctubs2") {
          const std::string referenceStr{
              "rMin=6.9551*m rMax=9*m cutAtStart=6.9551*m cutAtDelta=7.20045*m cutInside=true startPhi=0*deg "
              "deltaPhi=15*deg zHalf=6.57005*m"};
          char tubeParams[400];
          std::string boolVal{"false"};
          if (tubs.cutInside())
            boolVal = "true";
          (void)sprintf(tubeParams,
                        "rMin=%.5g*m rMax=%g*m cutAtStart=%.5g*m cutAtDelta=%.6g*m cutInside=%s startPhi=%g*deg "
                        "deltaPhi=%g*deg zHalf=%.6g*m",
                        convertCmToM(tubs.rIn()),
                        convertCmToM(tubs.rOut()),
                        convertCmToM(tubs.cutAtStart()),
                        convertCmToM(tubs.cutAtDelta()),
                        boolVal.c_str(),
                        convertRadToDeg(tubs.startPhi()),
                        convertRadToDeg(tubs.deltaPhi()),
                        convertCmToM(tubs.zHalf()));
          if (referenceStr == tubeParams) {
            std::cout << "Correct match with reference: " << referenceStr << std::endl;
          } else {
            std::cout << "ERROR: Mismatch of params for tube segment." << std::endl;
            std::cout << tubeParams << " does not match reference " << std::endl;
            std::cout << referenceStr << std::endl;
            CPPUNIT_ASSERT(referenceStr == tubeParams);
          }
        }
        break;
      }
      case cms::DDSolidShape::ddextrudedpolygon: {
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
        if (fview.name() == "extrudedpgon") {
          std::string referenceStr{"<XYPoint x=-30*cm y=-31*cm/>"};
          referenceStr += "\n<XYPoint x=-29*cm y=30*cm/>";
          referenceStr += "\n<XYPoint x=30.5*cm y=29.5*cm/>";
          referenceStr += "\n<XYPoint x=30.2*cm y=-30.1*cm/>";
          referenceStr += "\n<XYPoint x=15*cm y=-30.2*cm/>";
          referenceStr += "\n<XYPoint x=14.9*cm y=15.1*cm/>";
          referenceStr += "\n<XYPoint x=-15*cm y=14.8*cm/>";
          referenceStr += "\n<XYPoint x=-15.1*cm y=-29.8*cm/>";
          referenceStr += "\n<ZXYSection z=-60*cm x=0*cm y=30*cm scale=0.8/>";
          referenceStr += "\n<ZXYSection z=-14.9*cm x=0.1*cm y=-30*cm scale=1/>";
          referenceStr += "\n<ZXYSection z=10*cm x=0.2*cm y=0.4*cm scale=0.6/>";
          referenceStr += "\n<ZXYSection z=60*cm x=0.3*cm y=29.9*cm scale=1.2/>\n";
          char shapeParams[2400];
          shapeParams[0] = '\0';
          char fullLine[12][200];
          for (int index = 0; index < 8; ++index) {
            (void)sprintf(fullLine[index], "<XYPoint x=%g*cm ", pgon.xVec()[index + 8]);
          }
          for (int index = 0; index < 8; ++index) {
            char line[100];
            (void)sprintf(line, "y=%g*cm/>\n", pgon.yVec()[index + 8]);
            (void)strcat(fullLine[index], line);
          }
          for (int index = 8; index < 12; ++index) {
            (void)sprintf(fullLine[index], "<ZXYSection z=%g*cm ", pgon.zVec()[index - 8]);
          }
          for (int index = 8; index < 12; ++index) {
            char line[100];
            (void)sprintf(line, "x=%g*cm ", pgon.zxVec()[index - 4]);
            (void)strcat(fullLine[index], line);
          }
          for (int index = 8; index < 12; ++index) {
            char line[100];
            (void)sprintf(line, "y=%g*cm ", pgon.zyVec()[index - 4]);
            (void)strcat(fullLine[index], line);
          }
          for (int index = 8; index < 12; ++index) {
            char line[100];
            (void)sprintf(line, "scale=%g/>\n", pgon.zscaleVec()[index - 4]);
            (void)strcat(fullLine[index], line);
          }
          for (int index = 0; index < 12; ++index) {
            (void)strcat(shapeParams, fullLine[index]);
          }
          if (referenceStr == shapeParams) {
            std::cout << "Correct match with reference:\n" << referenceStr << std::endl;
          } else {
            std::cout << "ERROR: Mismatch of params for extruded polygon." << std::endl;
            std::cout << shapeParams << " does not match reference " << std::endl;
            std::cout << referenceStr << std::endl;
            CPPUNIT_ASSERT(referenceStr == shapeParams);
          }
        }
        break;
      }
      default:
        cout << "Shape not yet supported." << std::endl;
        break;
    }
  }
}
