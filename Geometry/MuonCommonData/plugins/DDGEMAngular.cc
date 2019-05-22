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
#include "DataFormats/Math/interface/GeantUnits.h"
#include "Geometry/MuonCommonData/plugins/DDGEMAngular.h"

using namespace geant_units::operators;

//#define EDM_ML_DEBUG

DDGEMAngular::DDGEMAngular() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "DDGEMAngular: Creating an instance";
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
  edm::LogVerbatim("MuonGeom") 
    << "DDGEMAngular: Parameters for positioning-- " << n 
    << " copies in steps of " << convertRadToDeg( stepAngle )
    << " from " << convertRadToDeg( startAngle )
    << " (inversion flag " << invert << ") \trPos " << rPos
    << " Zoffest " << zoffset << "\tStart and inremental "
    << "copy nos " << startCopyNo << ", " << incrCopyNo;
#endif

  rotns       = sArgs["RotNameSpace"];
  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") 
    << "DDGEMAngular: Parent " << parent().name() << "\tChild " << childName 
    << "\tNameSpace " << idNameSpace << "\tRotation Namespace " << rotns;
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
      edm::LogVerbatim("MuonGeom") 
	<< "DDGEMAngular: Creating a new rotation "
	<< DDName(rotstr, idNameSpace) << "\t " 
	<< convertRadToDeg( thetax ) << ", " << convertRadToDeg(  phix ) <<", "
	<< convertRadToDeg(  thetay ) << ", 0, " << convertRadToDeg( thetax )
	<< ", " << convertRadToDeg(  phiz );
#endif
      rotation = DDrot(DDName(rotstr, rotns), thetax, phix, thetay, 0., thetax, phiz);
    } 
    
    DDTranslation tran(rPos*cos(phitmp), rPos*sin(phitmp), zoffset);
  
    DDName parentName = parent().name(); 
    cpv.position(DDName(childName,idNameSpace), parentName, copyNo, tran, rotation);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MuonGeom") 
      << "DDGEMAngular: " << DDName(childName, idNameSpace) << " number " 
      << copyNo << " positioned in " << parentName 
      << " at " << tran << " with " << rotstr << " " << rotation;
#endif
    phi    += stepAngle;
    copyNo += incrCopyNo;
  }
}
