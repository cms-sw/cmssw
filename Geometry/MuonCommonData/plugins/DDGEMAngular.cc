///////////////////////////////////////////////////////////////////////////////
// File: DDGEMAngular.cc
// Description: Position inside the mother according to (eta,phi) 
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "Geometry/MuonCommonData/plugins/DDGEMAngular.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDGEMAngular::DDGEMAngular() {
  edm::LogInfo("MuonGeom") << "DDGEMAngular test: Creating an instance";
}

DDGEMAngular::~DDGEMAngular() {}

void DDGEMAngular::initialize(const DDNumericArguments & nArgs,
			       const DDVectorArguments & ,
			       const DDMapArguments & ,
			       const DDStringArguments & sArgs,
			       const DDStringVectorArguments & ) {

  startAngle  = nArgs["startAngle"];
  stepAngle   = nArgs["stepAngle"];
  invert      = int (nArgs["invert"]);
  rPos        = nArgs["rPosition"];
  zoffset     = nArgs["zoffset"];
  n           = int (nArgs["n"]);
  startCopyNo = int (nArgs["startCopyNo"]);
  incrCopyNo  = int (nArgs["incrCopyNo"]);
  edm::LogInfo("MuonGeom") << "DDGEMAngular debug: Parameters for positioning-- "
		       << n << " copies in steps of " << stepAngle/CLHEP::deg 
		       << " from " << startAngle/CLHEP::deg 
		       << " (inversion flag " << invert << ") \trPos " << rPos
		       << " Zoffest " << zoffset << "\tStart and inremental "
		       << "copy nos " << startCopyNo << ", " << incrCopyNo
		       << std::endl;

  rotns       = sArgs["RotNameSpace"];
  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name(); 
  edm::LogInfo("MuonGeom") << "DDGEMAngular debug: Parent " << parentName 
		       << "\tChild " << childName << "\tNameSpace "
		       << idNameSpace << "\tRotation Namespace " << rotns << std::endl;
}

void DDGEMAngular::execute(DDCompactView& cpv) {

  double phi    = startAngle;
  int    copyNo = startCopyNo;

  for (int ii=0; ii<n; ii++) {

    double phideg = phi/CLHEP::deg;
    int    iphi;
    if (phideg > 0)  iphi = int(phideg+0.1);
    else             iphi = int(phideg-0.1);
    if (iphi >= 360) iphi   -= 360;
    phideg = iphi;
    DDRotation rotation;
    std::string rotstr("NULL");

    rotstr = "RG"; 
    if (invert > 0)                rotstr += "I";
    if (phideg >=0 && phideg < 10) rotstr += "00"; 
    else if (phideg < 100)         rotstr += "0";
    rotstr  += dbl_to_string(phideg);
    rotation = DDRotation(DDName(rotstr, rotns)); 
    if (!rotation) {
      double thetax = 90.0;
      double phix   = invert==0 ? (90.0+phideg) : (-90.0+phideg);
      double thetay = invert==0 ? 0.0 : 180.0;
      double phiz   = phideg;
      edm::LogInfo("MuonGeom") << "DDGEMAngular test: Creating a new rotation "
			     << DDName(rotstr, idNameSpace) << "\t " 
			     << thetax << ", " << phix << ", " << thetay
			     << ", 0, " << thetax << ", " << phiz << std::endl;
      rotation = DDrot(DDName(rotstr, rotns), thetax*CLHEP::deg, 
		       phix*CLHEP::deg, thetay*CLHEP::deg, 0*CLHEP::deg,
		       thetax*CLHEP::deg, phiz*CLHEP::deg);
    } 
    
    DDTranslation tran(rPos*cos(phideg*CLHEP::deg), rPos*sin(phideg*CLHEP::deg), zoffset);
  
    DDName parentName = parent().name(); 
    cpv.position(DDName(childName,idNameSpace), parentName, copyNo, tran, rotation);
    edm::LogInfo("MuonGeom") << "DDGEMAngular test: " 
			 << DDName(childName, idNameSpace) << " number " 
			 << copyNo << " positioned in " << parentName << " at "
			 << tran << " with " << rotstr << " " << rotation << "\n";
    phi    += stepAngle;
    copyNo += incrCopyNo;
  }
}
