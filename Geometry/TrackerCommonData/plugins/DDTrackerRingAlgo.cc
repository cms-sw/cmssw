///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerRingAlgo.cc
// Description:  Tilts and positions n copies of a module at prescribed phi
// values within a ring. The module can also be flipped if requested.
///////////////////////////////////////////////////////////////////////////////


#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTrackerRingAlgo.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDTrackerRingAlgo::DDTrackerRingAlgo() {
  LogDebug("TrackerGeom") << "DDTrackerRingAlgo info: Creating an instance";
}

DDTrackerRingAlgo::~DDTrackerRingAlgo() {}

void DDTrackerRingAlgo::initialize(const DDNumericArguments & nArgs,
				  const DDVectorArguments & vArgs,
				  const DDMapArguments & ,
				  const DDStringArguments & sArgs,
				  const DDStringVectorArguments & ) {

  n             = int(nArgs["N"]);
  startCopyNo   = int(nArgs["StartCopyNo"]);
  incrCopyNo    = int(nArgs["IncrCopyNo"]);
  rangeAngle    = nArgs["RangeAngle"];
  startAngle    = nArgs["StartAngle"];
  radius        = nArgs["Radius"];
  center        = vArgs["Center"];
  isZPlus       = bool(nArgs["IsZPlus"]);
  tiltAngle     = nArgs["TiltAngle"];
  isFlipped     = bool(nArgs["IsFlipped"]);
  
  if (fabs(rangeAngle-360.0*CLHEP::deg)<0.001*CLHEP::deg) {
    delta    =   rangeAngle/double(n);
  } else {
    if (n > 1) {
      delta    =   rangeAngle/double(n-1);
    } else {
      delta = 0.;
    }
  }  

  LogDebug("TrackerGeom") << "DDTrackerRingAlgo debug: Parameters for position"
			  << "ing:: n " << n << " Start, Range, Delta "
			  << startAngle/CLHEP::deg << " "
			  << rangeAngle/CLHEP::deg << " " << delta/CLHEP::deg
			  << " Radius " << radius << " Centre " << center[0]
			  << ", " << center[1] << ", "<<center[2];

  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"];

  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDTrackerRingAlgo debug: Parent " << parentName
			  << "\tChild " << childName << " NameSpace "
			  << idNameSpace;
}

void DDTrackerRingAlgo::execute(DDCompactView& cpv) {

  DDRotation flipRot, tiltRot, phiRot, globalRot; // Identity
  DDRotationMatrix flipMatrix, tiltMatrix, phiRotMatrix, globalRotMatrix; // Identity matrix
  std::string rotstr = "RTrackerRingAlgo";

  // flipMatrix calculus
  if (isFlipped) {
    std::string flipRotstr = rotstr + "Flip";
    flipRot = DDRotation(DDName(flipRotstr, idNameSpace));
    if (!flipRot) {
      LogDebug("TrackerGeom") << "DDTrackerRingAlgo test: Creating a new rotation: " << flipRotstr
			      << "\t90., 180., "
			      << "90., 90., "
			      << "180., 0.";
      flipRot = DDrot(DDName(flipRotstr, idNameSpace), 
		      90.*CLHEP::deg, 180.*CLHEP::deg, 90.*CLHEP::deg, 90.*CLHEP::deg, 180.*CLHEP::deg, 0.);
    }
    flipMatrix = *flipRot.matrix();
  }
  // tiltMatrix calculus
  if (isZPlus) {
    std::string tiltRotstr = rotstr + "Tilt" + dbl_to_string(tiltAngle/CLHEP::deg) + "ZPlus";
    tiltRot = DDRotation(DDName(tiltRotstr, idNameSpace));
    if (!tiltRot) {
      LogDebug("TrackerGeom") << "DDTrackerRingAlgo test: Creating a new rotation: " << tiltRotstr
			      << "\t90., 90., "
			      << tiltAngle/CLHEP::deg << ", 180., "
			      << 90. - tiltAngle/CLHEP::deg << ", 0.";
      tiltRot = DDrot(DDName(tiltRotstr, idNameSpace), 
		      90.*CLHEP::deg, 90.*CLHEP::deg, tiltAngle, 180.*CLHEP::deg, 90.*CLHEP::deg - tiltAngle, 0.);
    }
    tiltMatrix = *tiltRot.matrix();
    if (isFlipped) { tiltMatrix *= flipMatrix; }
  }
  else {
    std::string tiltRotstr = rotstr + "Tilt" + dbl_to_string(tiltAngle/CLHEP::deg) + "ZMinus";
    tiltRot = DDRotation(DDName(tiltRotstr, idNameSpace));
    if (!tiltRot) {
      LogDebug("TrackerGeom") << "DDTrackerRingAlgo test: Creating a new rotation: " << tiltRotstr
			      << "\t90., 90., "
			      << tiltAngle/CLHEP::deg << ", 0., "
			      << 90. + tiltAngle/CLHEP::deg << ", 0.";
      tiltRot = DDrot(DDName(tiltRotstr, idNameSpace), 
		      90.*CLHEP::deg, 90.*CLHEP::deg, tiltAngle, 0., 90.*CLHEP::deg + tiltAngle, 0.);
    }
    tiltMatrix = *tiltRot.matrix();
    if (isFlipped) { tiltMatrix *= flipMatrix; }
  }

  // Loops for all phi values
  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second); 
  double theta  = 90.*CLHEP::deg;
  int    copy   = startCopyNo;
  double phi    = startAngle;

  for (int i=0; i<n; i++) {

    // phiRotMatrix calculus
    double phix = phi;
    double phiy = phix + 90.*CLHEP::deg;
    double phideg = phix/CLHEP::deg;  
    if (phideg != 0) {
      std::string phiRotstr = rotstr + "Phi" + dbl_to_string(phideg*10.);
      phiRot = DDRotation(DDName(phiRotstr, idNameSpace));
      if (!phiRot) {
	LogDebug("TrackerGeom") << "DDTrackerRingAlgo test: Creating a new rotation: " << phiRotstr
				<< "\t90., " << phix/CLHEP::deg
				<< ", 90.," << phiy/CLHEP::deg
				<<", 0., 0.";
	phiRot = DDrot(DDName(phiRotstr, idNameSpace), theta, phix, theta, phiy, 0., 0.);
      }
      phiRotMatrix = *phiRot.matrix();
    }

    // globalRot def
    std::string globalRotstr = rotstr + "Phi" + dbl_to_string(phideg*10.) + "Tilt" + dbl_to_string(tiltAngle/CLHEP::deg);
    if (isZPlus) {
      globalRotstr += "ZPlus";
      if (isFlipped) { globalRotstr += "Flip"; }
    }
    else { 
      globalRotstr += "ZMinus"; 
      if (isFlipped) { globalRotstr += "Flip"; }
    }
    globalRot = DDRotation(DDName(globalRotstr, idNameSpace));
    if (!globalRot) {
      LogDebug("TrackerGeom") << "DDTrackerRingAlgo test: Creating a new "
			      << "rotation: " << globalRotstr;
      globalRotMatrix = phiRotMatrix * tiltMatrix;
      globalRot = DDrot(DDName(globalRotstr, idNameSpace), new DDRotationMatrix(globalRotMatrix));
    }
   
    // translation def
    double xpos = radius*cos(phi) + center[0];
    double ypos = radius*sin(phi) + center[1];
    double zpos = center[2];
    DDTranslation tran(xpos, ypos, zpos);
  
    // Positions child with respect to parent
    cpv.position(child, mother, copy, tran, globalRot);
    LogDebug("TrackerGeom") << "DDTrackerRingAlgo test " << child << " number "
			    << copy << " positioned in " << mother << " at "
			    << tran  << " with " << globalRot;

    copy += incrCopyNo;
    phi  += delta;
  }
}
