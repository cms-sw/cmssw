///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerRingAlgo.cc
// Description:  Tilts and positions n copies of a module at prescribed phi
// values within a ring. The module can also be flipped if requested.
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <string>
#include <vector>

/*
  Tilts and positions n copies of a module at prescribed phi values
  within a ring. The module can also be flipped if requested.

  (radius, Phi, Z) refers to the cylindrical coordinates in the global frame of reference.
  
  A module's tilt angle is defined with respect to the global frame of reference's Z axis.
  Example, in the outer tracker : For a straight barrel module, tiltAngle = 0°.
  For a module in the endcaps, tiltAngle = 90°.
  tiltAngle ∈ [0, 90°].
  Please note that parameter tiltAngle has to be set regardless of any sign consideration,
  to the absolute value of the module's tilt angle.

  == Example of use : ==

  <Algorithm name="track:DDTrackerRingAlgo">
  <rParent name="tracker:Ring5Layer1Plus"/>
  <String name="ChildName" value="tracker:BModule5Layer1"/>
  <Numeric name="N" value="9"/>
  <Numeric name="StartCopyNo" value="1"/>
  <Numeric name="IncrCopyNo" value="2"/>
  <Numeric name="RangeAngle" value="360*deg"/>
  <Numeric name="StartAngle" value="90*deg"/>
  <Numeric name="Radius" value="247"/>
  <Vector name="Center" type="numeric" nEntries="3">0,0,-5.45415</Vector>
  <Numeric name="IsZPlus" value="1"/>
  <Numeric name="TiltAngle" value="47*deg"/>
  <Numeric name="IsFlipped" value="1"/>
  </Algorithm>
*/

using namespace std;

class DDTrackerRingAlgo : public DDAlgorithm {
public:
  // Constructor and Destructor
  DDTrackerRingAlgo();
  ~DDTrackerRingAlgo() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  int n;                  //Number of copies
  int startCopyNo;        //Start Copy number
  int incrCopyNo;         //Increment in Copy number
  double rangeAngle;      //Range in Phi angle
  double startAngle;      //Start Phi angle
  double radius;          //Radius
  vector<double> center;  //Phi values
  bool isZPlus;           //Is Z positive ?
  double tiltAngle;       //Module's tilt angle (absolute value)
  bool isFlipped;         //Is the module flipped ?
  double delta;           //Increment in Phi

  string idNameSpace;  //Namespace of this and ALL sub-parts
  string childName;    //Child name
};

DDTrackerRingAlgo::DDTrackerRingAlgo() { LogDebug("TrackerGeom") << "DDTrackerRingAlgo info: Creating an instance"; }

DDTrackerRingAlgo::~DDTrackerRingAlgo() {}

void DDTrackerRingAlgo::initialize(const DDNumericArguments& nArgs,
                                   const DDVectorArguments& vArgs,
                                   const DDMapArguments&,
                                   const DDStringArguments& sArgs,
                                   const DDStringVectorArguments&) {
  n = int(nArgs["N"]);
  startCopyNo = int(nArgs["StartCopyNo"]);
  incrCopyNo = int(nArgs["IncrCopyNo"]);
  rangeAngle = nArgs["RangeAngle"];
  startAngle = nArgs["StartAngle"];
  radius = nArgs["Radius"];
  center = vArgs["Center"];
  isZPlus = bool(nArgs["IsZPlus"]);
  tiltAngle = nArgs["TiltAngle"];
  isFlipped = bool(nArgs["IsFlipped"]);

  if (fabs(rangeAngle - 360.0 * CLHEP::deg) < 0.001 * CLHEP::deg) {
    delta = rangeAngle / double(n);
  } else {
    if (n > 1) {
      delta = rangeAngle / double(n - 1);
    } else {
      delta = 0.;
    }
  }

  LogDebug("TrackerGeom") << "DDTrackerRingAlgo debug: Parameters for position"
                          << "ing:: n " << n << " Start, Range, Delta " << startAngle / CLHEP::deg << " "
                          << rangeAngle / CLHEP::deg << " " << delta / CLHEP::deg << " Radius " << radius << " Centre "
                          << center[0] << ", " << center[1] << ", " << center[2];

  idNameSpace = DDCurrentNamespace::ns();
  childName = sArgs["ChildName"];

  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDTrackerRingAlgo debug: Parent " << parentName << "\tChild " << childName
                          << " NameSpace " << idNameSpace;
}

void DDTrackerRingAlgo::execute(DDCompactView& cpv) {
  DDRotation flipRot, tiltRot, phiRot, globalRot;                          // Identity
  DDRotationMatrix flipMatrix, tiltMatrix, phiRotMatrix, globalRotMatrix;  // Identity matrix
  string rotstr = "RTrackerRingAlgo";

  // flipMatrix calculus
  if (isFlipped) {
    string flipRotstr = rotstr + "Flip";
    flipRot = DDRotation(DDName(flipRotstr, idNameSpace));
    if (!flipRot) {
      LogDebug("TrackerGeom") << "DDTrackerRingAlgo test: Creating a new rotation: " << flipRotstr << "\t90., 180., "
                              << "90., 90., "
                              << "180., 0.";
      flipRot = DDrot(DDName(flipRotstr, idNameSpace),
                      90. * CLHEP::deg,
                      180. * CLHEP::deg,
                      90. * CLHEP::deg,
                      90. * CLHEP::deg,
                      180. * CLHEP::deg,
                      0.);
    }
    flipMatrix = flipRot.matrix();
  }
  // tiltMatrix calculus
  if (isZPlus) {
    string tiltRotstr = rotstr + "Tilt" + to_string(tiltAngle / CLHEP::deg) + "ZPlus";
    tiltRot = DDRotation(DDName(tiltRotstr, idNameSpace));
    if (!tiltRot) {
      LogDebug("TrackerGeom") << "DDTrackerRingAlgo test: Creating a new rotation: " << tiltRotstr << "\t90., 90., "
                              << tiltAngle / CLHEP::deg << ", 180., " << 90. - tiltAngle / CLHEP::deg << ", 0.";
      tiltRot = DDrot(DDName(tiltRotstr, idNameSpace),
                      90. * CLHEP::deg,
                      90. * CLHEP::deg,
                      tiltAngle,
                      180. * CLHEP::deg,
                      90. * CLHEP::deg - tiltAngle,
                      0.);
    }
    tiltMatrix = tiltRot.matrix();
    if (isFlipped) {
      tiltMatrix *= flipMatrix;
    }
  } else {
    string tiltRotstr = rotstr + "Tilt" + to_string(tiltAngle / CLHEP::deg) + "ZMinus";
    tiltRot = DDRotation(DDName(tiltRotstr, idNameSpace));
    if (!tiltRot) {
      LogDebug("TrackerGeom") << "DDTrackerRingAlgo test: Creating a new rotation: " << tiltRotstr << "\t90., 90., "
                              << tiltAngle / CLHEP::deg << ", 0., " << 90. + tiltAngle / CLHEP::deg << ", 0.";
      tiltRot = DDrot(DDName(tiltRotstr, idNameSpace),
                      90. * CLHEP::deg,
                      90. * CLHEP::deg,
                      tiltAngle,
                      0.,
                      90. * CLHEP::deg + tiltAngle,
                      0.);
    }
    tiltMatrix = tiltRot.matrix();
    if (isFlipped) {
      tiltMatrix *= flipMatrix;
    }
  }

  // Loops for all phi values
  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  double theta = 90. * CLHEP::deg;
  int copy = startCopyNo;
  double phi = startAngle;

  for (int i = 0; i < n; i++) {
    // phiRotMatrix calculus
    double phix = phi;
    double phiy = phix + 90. * CLHEP::deg;
    double phideg = phix / CLHEP::deg;
    if (phideg != 0) {
      string phiRotstr = rotstr + "Phi" + to_string(phideg * 10.);
      phiRot = DDRotation(DDName(phiRotstr, idNameSpace));
      if (!phiRot) {
        LogDebug("TrackerGeom") << "DDTrackerRingAlgo test: Creating a new rotation: " << phiRotstr << "\t90., "
                                << phix / CLHEP::deg << ", 90.," << phiy / CLHEP::deg << ", 0., 0.";
        phiRot = DDrot(DDName(phiRotstr, idNameSpace), theta, phix, theta, phiy, 0., 0.);
      }
      phiRotMatrix = phiRot.matrix();
    }

    // globalRot def
    string globalRotstr = rotstr + "Phi" + to_string(phideg * 10.) + "Tilt" + to_string(tiltAngle / CLHEP::deg);
    if (isZPlus) {
      globalRotstr += "ZPlus";
      if (isFlipped) {
        globalRotstr += "Flip";
      }
    } else {
      globalRotstr += "ZMinus";
      if (isFlipped) {
        globalRotstr += "Flip";
      }
    }
    globalRot = DDRotation(DDName(globalRotstr, idNameSpace));
    if (!globalRot) {
      LogDebug("TrackerGeom") << "DDTrackerRingAlgo test: Creating a new "
                              << "rotation: " << globalRotstr;
      globalRotMatrix = phiRotMatrix * tiltMatrix;
      globalRot = DDrot(DDName(globalRotstr, idNameSpace), make_unique<DDRotationMatrix>(globalRotMatrix));
    }

    // translation def
    double xpos = radius * cos(phi) + center[0];
    double ypos = radius * sin(phi) + center[1];
    double zpos = center[2];
    DDTranslation tran(xpos, ypos, zpos);

    // Positions child with respect to parent
    cpv.position(child, mother, copy, tran, globalRot);
    LogDebug("TrackerGeom") << "DDTrackerRingAlgo test " << child << " number " << copy << " positioned in " << mother
                            << " at " << tran << " with " << globalRot;

    copy += incrCopyNo;
    phi += delta;
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDTrackerRingAlgo, "track:DDTrackerRingAlgo");
