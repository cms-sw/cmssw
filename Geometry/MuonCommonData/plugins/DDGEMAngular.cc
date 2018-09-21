///////////////////////////////////////////////////////////////////////////////
// File: DDGEMAngular.cc
// Description: Position inside the mother according to (eta,phi) 
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDUnits.h"
#include "Geometry/MuonCommonData/plugins/DDGEMAngular.h"

using namespace dd;
using namespace dd::operators;

//#define EDM_ML_DEBUG

DDGEMAngular::DDGEMAngular() {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("MuonGeom") << "DDGEMAngular test: Creating an instance";
#endif
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
#ifdef EDM_ML_DEBUG
  edm::LogInfo("MuonGeom") << "DDGEMAngular debug: Parameters for positioning-- "
			   << n << " copies in steps of " << CONVERT_TO( stepAngle, deg )
			   << " from " << CONVERT_TO( startAngle, deg )
			   << " (inversion flag " << invert << ") \trPos " << rPos
			   << " Zoffest " << zoffset << "\tStart and inremental "
			   << "copy nos " << startCopyNo << ", " << incrCopyNo;
#endif

  rotns       = sArgs["RotNameSpace"];
  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
#ifdef EDM_ML_DEBUG
  DDName parentName = parent().name(); 
  edm::LogInfo("MuonGeom") << "DDGEMAngular debug: Parent " << parentName 
			   << "\tChild " << childName << "\tNameSpace "
			   << idNameSpace << "\tRotation Namespace " << rotns;
#endif
}

void DDGEMAngular::execute(DDCompactView& cpv) {

  double phi    = startAngle;
  int    copyNo = startCopyNo;

  for (int ii=0; ii<n; ii++) {

    double phitmp = phi;
    if (phitmp >= 2._pi) phitmp -= 2._pi;
    DDRotation rotation;
    std::string rotstr("RG");

    if (invert > 0) rotstr += "I";
    rotstr  += formatAsDegrees(phitmp);
    rotation = DDRotation(DDName(rotstr, rotns)); 
    if (!rotation) {
      double thetax = 90.0_deg;
      double phix   = invert==0 ? (90.0_deg + phitmp) : (-90.0_deg + phitmp);
      double thetay = invert==0 ? 0.0 : 180.0_deg;
      double phiz   = phitmp;
#ifdef EDM_ML_DEBUG
      edm::LogInfo("MuonGeom") << "DDGEMAngular test: Creating a new rotation "
			       << DDName(rotstr, idNameSpace) << "\t " 
			       << CONVERT_TO( thetax, deg ) << ", " << CONVERT_TO( phix, deg ) << ", " << CONVERT_TO( thetay, deg )
			       << ", 0, " << CONVERT_TO( thetax, deg )<< ", " << CONVERT_TO( phiz, deg );
#endif
      rotation = DDrot(DDName(rotstr, rotns), thetax, phix, thetay, 0., thetax, phiz);
    } 
    
    DDTranslation tran(rPos*cos(phitmp), rPos*sin(phitmp), zoffset);
  
    DDName parentName = parent().name(); 
    cpv.position(DDName(childName,idNameSpace), parentName, copyNo, tran, rotation);
#ifdef EDM_ML_DEBUG
    edm::LogInfo("MuonGeom") << "DDGEMAngular test: " 
			     << DDName(childName, idNameSpace) << " number " 
			     << copyNo << " positioned in " << parentName 
			     << " at " << tran << " with " << rotstr << " "
			     << rotation;
#endif
    phi    += stepAngle;
    copyNo += incrCopyNo;
  }
}
