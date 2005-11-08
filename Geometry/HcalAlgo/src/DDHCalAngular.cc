///////////////////////////////////////////////////////////////////////////////
// File: DDHCalAngular.cc
// Description: Position inside the mother according to (eta,phi) 
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HcalAlgo/interface/DDHCalAngular.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

DDHCalAngular::DDHCalAngular() {
  DCOUT('a', "DDHCalAngular test: Creating an instance");
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
  DCOUT('A', "DDHCalAngular debug: Parameters for positioning-- " << n << " copies in " << rangeAngle/deg << " from " << startAngle/deg << "\tShifts " << shiftX << ", " << shiftY << " along x, y axes; \tZoffest " << zoffset << "\tStart and inremental copy nos " << startCopyNo << ", " << incrCopyNo);

  rotns       = sArgs["RotNameSpace"];
  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name(); 
  DCOUT('A', "DDHCalAngular debug: Parent " << parentName << "\tChild " << childName << "\tNameSpace " << idNameSpace << "\tRotation Namespace " << rotns);
}

void DDHCalAngular::execute() {

  double dphi   = rangeAngle/n;
  double phi    = startAngle;
  int    copyNo = startCopyNo;

  for (int ii=0; ii<n; ii++) {

    double phideg = phi/deg;
    int    iphi   = int(phideg+0.1);
    if (iphi >= 360) iphi   -= 360;
    phideg = iphi;
    DDRotation rotation;
    string rotstr("NULL");

    if (iphi != 0) {
      rotstr = "R"; 
      if (phideg < 100) rotstr = "R0"; 
      rotstr = rotstr + dbl_to_string(phideg);
      rotation = DDRotation(DDName(rotstr, rotns)); 
      if (!rotation) {
        DCOUT('a', "DDHCalAngular test: Creating a new rotation " << DDName(rotstr, idNameSpace) << "\t" << 90 << "," << phideg << ","  << 90 << "," << (phideg+90) << "," << 0 << "," << 0);
        rotation = DDrot(DDName(rotstr, idNameSpace), 90*deg, phideg*deg, 
                         90*deg, (90+phideg)*deg, 0*deg,  0*deg);
      } 
    }
    
    double xpos = shiftX*cos(phi) - shiftY*sin(phi);
    double ypos = shiftX*sin(phi) + shiftY*cos(phi);
    DDTranslation tran(xpos, ypos, zoffset);
  
    DDName parentName = parent().name(); 
    DDpos (DDName(childName,idNameSpace), parentName, copyNo, tran, rotation);
    DCOUT('a', "DDHCalAngular test: " << DDName(childName, idNameSpace) << " number " << copyNo << " positioned in " << parentName << " at " << tran << " with " << rotation);
    phi    += dphi;
    copyNo += incrCopyNo;
  }
}
