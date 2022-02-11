///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerIrregularRingAlgo.cc
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
#include "DataFormats/Math/interface/angle_units.h"

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

  <Algorithm name="track:DDTrackerIrregularRingAlgo">
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
using namespace angle_units::operators;

class DDTrackerIrregularRingAlgo : public DDAlgorithm {
public:
  // Constructor and Destructor
  DDTrackerIrregularRingAlgo();
  ~DDTrackerIrregularRingAlgo() override;

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
  vector<double> phiAngles;
  vector<double> radiusValues;
  vector<double> yawAngles;
  bool isZPlus;      //Is Z positive ?
  double tiltAngle;  //Module's tilt angle (absolute value)
  bool isFlipped;    //Is the module flipped ?

  string idNameSpace;  //Namespace of this and ALL sub-parts
  string childName;    //Child name
};

DDTrackerIrregularRingAlgo::DDTrackerIrregularRingAlgo() {
  LogDebug("TrackerGeom") << "DDTrackerIrregularRingAlgo info: Creating an instance";
}

DDTrackerIrregularRingAlgo::~DDTrackerIrregularRingAlgo() = default;

void DDTrackerIrregularRingAlgo::initialize(const DDNumericArguments& nArgs,
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
  yawAngles = vArgs["yawAngleValues"];
  phiAngles = vArgs["phiAngleValues"];
  radiusValues = vArgs["radiusValues"];
  isZPlus = bool(nArgs["IsZPlus"]);
  tiltAngle = nArgs["TiltAngle"];
  isFlipped = bool(nArgs["IsFlipped"]);

  LogDebug("TrackerGeom") << "DDTrackerIrregularRingAlgo debug: Parameters for position"
                          << "ing:: n " << n << " Start, Range " << convertRadToDeg(startAngle) << " "
                          << convertRadToDeg(rangeAngle) << " Radius " << radius << " Centre " << center[0] << ", "
                          << center[1] << ", " << center[2];

  idNameSpace = DDCurrentNamespace::ns();
  childName = sArgs["ChildName"];

  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDTrackerIrregularRingAlgo debug: Parent " << parentName << "\tChild " << childName
                          << " NameSpace " << idNameSpace;
}

void DDTrackerIrregularRingAlgo::execute(DDCompactView& cpv) {
  DDRotation flipRot, tiltRot, phiOwnAxisRot, phiRot, globalRot;                                // Identity
  DDRotationMatrix flipMatrix, tiltMatrix, phiOwnAxisRotMatrix, phiRotMatrix, globalRotMatrix;  // Identity matrix
  string rotstr = "RTrackerRingAlgo";

  // flipMatrix calculus
  if (isFlipped) {
    string flipRotstr = rotstr + "Flip";
    flipRot = DDRotation(DDName(flipRotstr, idNameSpace));
    if (!flipRot) {
      LogDebug("TrackerGeom") << "DDTrackerIrregularRingAlgo test: Creating a new rotation: " << flipRotstr
                              << "\t90., 180., "
                              << "90., 90., "
                              << "180., 0.";
      flipRot = DDrot(DDName(flipRotstr, idNameSpace), 90._deg, 180._deg, 90._deg, 90._deg, 180._deg, 0.);
    }
    flipMatrix = flipRot.matrix();
  }
  // tiltMatrix calculus
  if (isZPlus) {
    string tiltRotstr = rotstr + "Tilt" + to_string(convertRadToDeg(tiltAngle)) + "ZPlus";
    tiltRot = DDRotation(DDName(tiltRotstr, idNameSpace));
    if (!tiltRot) {
      LogDebug("TrackerGeom") << "DDTrackerIrregularRingAlgo test: Creating a new rotation: " << tiltRotstr
                              << "\t90., 90., " << convertRadToDeg(tiltAngle) << ", 180., "
                              << 90. - convertRadToDeg(tiltAngle) << ", 0.";
      tiltRot = DDrot(DDName(tiltRotstr, idNameSpace), 90._deg, 90._deg, tiltAngle, 180._deg, 90._deg - tiltAngle, 0.);
    }
    tiltMatrix = tiltRot.matrix();
    if (isFlipped) {
      tiltMatrix *= flipMatrix;
    }
  } else {
    string tiltRotstr = rotstr + "Tilt" + to_string(convertRadToDeg(tiltAngle)) + "ZMinus";
    tiltRot = DDRotation(DDName(tiltRotstr, idNameSpace));
    if (!tiltRot) {
      LogDebug("TrackerGeom") << "DDTrackerIrregularRingAlgo test: Creating a new rotation: " << tiltRotstr
                              << "\t90., 90., " << convertRadToDeg(tiltAngle) << ", 0., "
                              << 90. + convertRadToDeg(tiltAngle) << ", 0.";
      tiltRot = DDrot(DDName(tiltRotstr, idNameSpace), 90._deg, 90._deg, tiltAngle, 0., 90._deg + tiltAngle, 0.);
    }
    tiltMatrix = tiltRot.matrix();
    if (isFlipped) {
      tiltMatrix *= flipMatrix;
    }
  }

  // Loops for all phi values
  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  double theta = 90._deg;
  int copy = startCopyNo;
  //double phi = startAngle;

  for (int i = 0; i < n; i++) {
    // phiRotMatrix calculus
    //double phix = phi;
    //double phix_ownaxis = 0._deg;
    double phix = convertDegToRad(phiAngles.at(i));
    double phix_ownaxis = convertDegToRad(yawAngles.at(i));
    radius = radiusValues.at(i);
    double phiy = phix + 90._deg;
    double phiy_ownaxis = phix_ownaxis + 90._deg;
    double phideg = convertRadToDeg(phix);
    double phideg_ownaxis = convertRadToDeg(phix_ownaxis);
    if (phideg_ownaxis != 0) {
      string phiOwnAxisRotstr = rotstr + "PhiOwnAxis" + to_string(phideg_ownaxis * 10.);
      phiOwnAxisRot = DDRotation(DDName(phiOwnAxisRotstr, idNameSpace));
      if (!phiOwnAxisRot) {
        LogDebug("TrackerGeom") << "DDTrackerIrregularRingAlgo test: Creating a new rotation: " << phiOwnAxisRotstr
                                << "\t90., " << convertRadToDeg(phix_ownaxis) << ", 90.,"
                                << convertRadToDeg(phiy_ownaxis) << ", 0., 0.";
        phiOwnAxisRot = DDrot(DDName(phiOwnAxisRotstr, idNameSpace), theta, phix_ownaxis, theta, phiy_ownaxis, 0., 0.);
      }
      phiOwnAxisRotMatrix = phiOwnAxisRot.matrix();
    }
    if (phideg != 0) {
      string phiRotstr = rotstr + "Phi" + to_string(phideg * 10.);
      phiRot = DDRotation(DDName(phiRotstr, idNameSpace));
      if (!phiRot) {
        LogDebug("TrackerGeom") << "DDTrackerIrregularRingAlgo test: Creating a new rotation: " << phiRotstr
                                << "\t90., " << convertRadToDeg(phix) << ", 90.," << convertRadToDeg(phiy)
                                << ", 0., 0.";
        phiRot = DDrot(DDName(phiRotstr, idNameSpace), theta, phix, theta, phiy, 0., 0.);
      }
      phiRotMatrix = phiRot.matrix();
    }

    // globalRot def
    string globalRotstr = rotstr + "Phi" + to_string(phideg * 10.) + "Tilt" + to_string(convertRadToDeg(tiltAngle));
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
      LogDebug("TrackerGeom") << "DDTrackerIrregularRingAlgo test: Creating a new "
                              << "rotation: " << globalRotstr;
      globalRotMatrix = phiOwnAxisRotMatrix * phiRotMatrix * tiltMatrix;
      globalRot = DDrot(DDName(globalRotstr, idNameSpace), make_unique<DDRotationMatrix>(globalRotMatrix));
    }

    // translation def
    double xpos = radius * cos(phix) + center[0];
    double ypos = radius * sin(phix) + center[1];
    double zpos = center[2];
    DDTranslation tran(xpos, ypos, zpos);

    // Positions child with respect to parent
    cpv.position(child, mother, copy, tran, globalRot);
    LogDebug("TrackerGeom") << "DDTrackerIrregularRingAlgo test " << child << " number " << copy << " positioned in "
                            << mother << " at " << tran << " with " << globalRot;

    copy += incrCopyNo;
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDTrackerIrregularRingAlgo, "track:DDTrackerIrregularRingAlgo");
