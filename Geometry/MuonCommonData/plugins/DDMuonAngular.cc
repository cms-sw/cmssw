///////////////////////////////////////////////////////////////////////////////
// File: DDMuonAngular.cc
// Description: Position inside the mother according to (eta,phi) 
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "Geometry/MuonCommonData/plugins/DDMuonAngular.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

DDMuonAngular::DDMuonAngular() {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("MuonGeom") << "DDMuonAngular test: Creating an instance";
#endif
}

DDMuonAngular::~DDMuonAngular() {}

void DDMuonAngular::initialize(const DDNumericArguments & nArgs,
			       const DDVectorArguments & ,
			       const DDMapArguments & ,
			       const DDStringArguments & sArgs,
			       const DDStringVectorArguments & ) {

  startAngle  = nArgs["startAngle"];
  stepAngle   = nArgs["stepAngle"];
  zoffset     = nArgs["zoffset"];
  n           = int (nArgs["n"]);
  startCopyNo = int (nArgs["startCopyNo"]);
  incrCopyNo  = int (nArgs["incrCopyNo"]);
#ifdef EDM_ML_DEBUG
  edm::LogInfo("MuonGeom") << "DDMuonAngular debug: Parameters for positioning-- "
			   << n << " copies in steps of " << stepAngle/CLHEP::deg 
			   << " from " << startAngle/CLHEP::deg << " \tZoffest " 
			   << zoffset << "\tStart and inremental copy nos " 
			   << startCopyNo << ", " << incrCopyNo;
#endif
  rotns       = sArgs["RotNameSpace"];
  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
#ifdef EDM_ML_DEBUG
  DDName parentName = parent().name(); 
  edm::LogInfo("MuonGeom") << "DDMuonAngular debug: Parent " << parentName 
			   << "\tChild " << childName << "\tNameSpace "
			   << idNameSpace << "\tRotation Namespace " << rotns;
#endif
}

void DDMuonAngular::execute(DDCompactView& cpv) {

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

    if (iphi != 0) {
      rotstr = "R"; 
      if (phideg >=0 && phideg < 10) rotstr = "R00"; 
      else if (phideg < 100)         rotstr = "R0";
      rotstr = rotstr + std::to_string(phideg);
      rotation = DDRotation(DDName(rotstr, rotns)); 
      if (!rotation) {
#ifdef EDM_ML_DEBUG
        edm::LogInfo("MuonGeom") << "DDMuonAngular test: Creating a new rotation "
				 << DDName(rotstr, idNameSpace) << "\t90, " 
				 << phideg << ", 90, " << (phideg+90) << ", 0, 0";
#endif
        rotation = DDrot(DDName(rotstr, rotns), 90*CLHEP::deg, 
			 phideg*CLHEP::deg, 90*CLHEP::deg, 
			 (90+phideg)*CLHEP::deg, 0*CLHEP::deg,  0*CLHEP::deg);
      } 
    }
    
    DDTranslation tran(0, 0, zoffset);
  
    DDName parentName = parent().name(); 
    cpv.position(DDName(childName,idNameSpace), parentName, copyNo, tran, rotation);
#ifdef EDM_ML_DEBUG
    edm::LogInfo("MuonGeom") << "DDMuonAngular test: " 
			     << DDName(childName, idNameSpace) << " number " 
			     << copyNo << " positioned in " << parentName 
			     << " at " << tran << " with " << rotstr << " " 
			     << rotation;
#endif
    phi    += stepAngle;
    copyNo += incrCopyNo;
  }
}
