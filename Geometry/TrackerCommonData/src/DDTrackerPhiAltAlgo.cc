///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerPhiAltAlgo.cc
// Description: Position n copies inside and outside at alternate phi values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerPhiAltAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTrackerPhiAltAlgo::DDTrackerPhiAltAlgo() {
  DCOUT('a', "DDTrackerPhiAltAlgo info: Creating an instance");
}

DDTrackerPhiAltAlgo::~DDTrackerPhiAltAlgo() {}

void DDTrackerPhiAltAlgo::initialize(const DDNumericArguments & nArgs,
				     const DDVectorArguments & ,
				     const DDMapArguments & ,
				     const DDStringArguments & sArgs,
				     const DDStringVectorArguments & ) {

  tilt       = nArgs["Tilt"];
  startAngle = nArgs["StartAngle"];
  rangeAngle = nArgs["RangeAngle"];
  radiusIn   = nArgs["RadiusIn"];
  radiusOut  = nArgs["RadiusOut"];
  zpos       = nArgs["ZPosition"];
  number     = int (nArgs["Number"]);
  startCopyNo= int (nArgs["StartCopyNo"]);
  incrCopyNo = int (nArgs["IncrCopyNo"]);

  DCOUT('A', "DDTrackerPhiAltAlgo debug: Parameters for positioning--" << " Tilt " << tilt << "\tStartAngle " << startAngle/deg << "\tRangeAngle " << rangeAngle/deg << "\tRin " << radiusIn << "\tRout " << radiusOut << "\t ZPos " << zpos << "\tCopy Numbers " << number << " Start/Increment " << startCopyNo << ", " << incrCopyNo);

  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name();
  DCOUT('A', "DDTrackerPhiAltAlgo debug: Parent " << parentName << "\tChild " << childName << " NameSpace " << idNameSpace);
}

void DDTrackerPhiAltAlgo::execute() {

  if (number > 0) {
    double theta  = 90.*deg;
    double dphi;
    if (number == 1 || fabs(rangeAngle-360.0*deg)<0.001*deg) 
      dphi = rangeAngle/number;
    else
      dphi = rangeAngle/(number-1);
    int copyNo = startCopyNo;

    DDName mother = parent().name();
    DDName child(DDSplit(childName).first, DDSplit(childName).second);
    for (int i=0; i<number; i++) {
      double phi  = startAngle + i*dphi;
      double phix = phi - tilt + 90.*deg;
      double phiy = phix + 90.*deg;
      double phideg = phix/deg;
  
      DDRotation rotation;
      if (phideg != 0) {
	string rotstr = DDSplit(childName).first + dbl_to_string(phideg*10.);
	rotation = DDRotation(DDName(rotstr, idNameSpace));
	if (!rotation) {
	  DCOUT('a', "DDTrackerPhiAltAlgo test: Creating a new rotation " << rotstr << "\t" << "90., " << phix/deg << ", 90.," << phiy/deg << ", 0, 0");
	  rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta,
			   phiy, 0., 0.);
	}
      }
	
      double xpos, ypos;
      if (i%2 == 0) {
	xpos = radiusIn*cos(phi);
	ypos = radiusIn*sin(phi);
      } else {
	xpos = radiusOut*cos(phi);
	ypos = radiusOut*sin(phi);
      }
      DDTranslation tran(xpos, ypos, zpos);
  
      DDpos (child, mother, copyNo, tran, rotation);
      DCOUT('a', "DDTrackerPhiAltAlgo test: " << child << " number " << copyNo << " positioned in " << mother << " at " << tran << " with " << rotation);
      copyNo += incrCopyNo;
    }
  }
}
