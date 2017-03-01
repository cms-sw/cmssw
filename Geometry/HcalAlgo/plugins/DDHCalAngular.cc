///////////////////////////////////////////////////////////////////////////////
// File: DDHCalAngular.cc
// Description: Position inside the mother according to (eta,phi) 
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "Geometry/HcalAlgo/plugins/DDHCalAngular.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDHCalAngular::DDHCalAngular() {
  LogDebug("HCalGeom") << "DDHCalAngular test: Creating an instance";
}

DDHCalAngular::~DDHCalAngular() {}

void DDHCalAngular::initialize(const DDNumericArguments & nArgs,
			       const DDVectorArguments & ,
			       const DDMapArguments & ,
			       const DDStringArguments & sArgs,
			       const DDStringVectorArguments & ) {

  startAngle  = nArgs["startAngle"];
  rangeAngle  = nArgs["rangeAngle"];
  shiftX      = nArgs["shiftX"];
  shiftY      = nArgs["shiftY"];
  zoffset     = nArgs["zoffset"];
  n           = int (nArgs["n"]);
  startCopyNo = int (nArgs["startCopyNo"]);
  incrCopyNo  = int (nArgs["incrCopyNo"]);
  LogDebug("HCalGeom") << "DDHCalAngular debug: Parameters for positioning-- "
		       << n << " copies in " << rangeAngle/CLHEP::deg 
		       << " from " << startAngle/CLHEP::deg << "\tShifts " 
		       << shiftX << ", " << shiftY 
		       << " along x, y axes; \tZoffest " << zoffset
		       << "\tStart and inremental copy nos " << startCopyNo 
		       << ", " << incrCopyNo;

  rotns       = sArgs["RotNameSpace"];
  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name(); 
  LogDebug("HCalGeom") << "DDHCalAngular debug: Parent " << parentName 
		       << "\tChild " << childName << "\tNameSpace "
		       << idNameSpace << "\tRotation Namespace " << rotns;
}

void DDHCalAngular::execute(DDCompactView& cpv) {

  double dphi   = rangeAngle/n;
  double phi    = startAngle;
  int    copyNo = startCopyNo;

  for (int ii=0; ii<n; ii++) {

    double phideg = phi/CLHEP::deg;
    int    iphi;
    if (phideg > 0) iphi = int(phideg+0.1);
    else            iphi = int(phideg-0.1);
    if (iphi >= 360) iphi   -= 360;
    phideg = iphi;
    DDRotation rotation;
    std::string rotstr("NULL");

    if (iphi != 0) {
      rotstr = "R"; 
      if (phideg >=0 && phideg < 100) rotstr = "R0"; 
      rotstr = rotstr + std::to_string(phideg);
      rotation = DDRotation(DDName(rotstr, rotns)); 
      if (!rotation) {
        LogDebug("HCalGeom") << "DDHCalAngular test: Creating a new rotation "
			     << DDName(rotstr, idNameSpace) << "\t90, " 
			     << phideg << ", 90, " << (phideg+90) << ", 0, 0";
        rotation = DDrot(DDName(rotstr, rotns), 90*CLHEP::deg, 
			 phideg*CLHEP::deg, 90*CLHEP::deg, 
			 (90+phideg)*CLHEP::deg, 0*CLHEP::deg,  0*CLHEP::deg);
      } 
    }
    
    double xpos = shiftX*cos(phi) - shiftY*sin(phi);
    double ypos = shiftX*sin(phi) + shiftY*cos(phi);
    DDTranslation tran(xpos, ypos, zoffset);
  
    DDName parentName = parent().name(); 
   cpv.position(DDName(childName,idNameSpace), parentName, copyNo, tran, rotation);
    LogDebug("HCalGeom") << "DDHCalAngular test: " 
			 << DDName(childName, idNameSpace) << " number " 
			 << copyNo << " positioned in " << parentName << " at "
			 << tran << " with " << rotation;
    phi    += dphi;
    copyNo += incrCopyNo;
  }
}
