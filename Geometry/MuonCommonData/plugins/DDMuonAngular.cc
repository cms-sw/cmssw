///////////////////////////////////////////////////////////////////////////////
// File: DDMuonAngular.cc
// Description: Position inside the mother according to (eta,phi) 
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "Geometry/MuonCommonData/plugins/DDMuonAngular.h"

using namespace geant_units::operators;

//#define EDM_ML_DEBUG

DDMuonAngular::DDMuonAngular() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "DDMuonAngular: Creating an instance";
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
  edm::LogVerbatim("MuonGeom") 
    << "DDMuonAngular: Parameters for positioning-- "  << n 
    << " copies in steps of " << convertRadToDeg( stepAngle )
    << " from " << convertRadToDeg( startAngle ) << " \tZoffest "
    << zoffset << "\tStart and inremental copy nos " 
    << startCopyNo << ", " << incrCopyNo;
#endif
  rotns       = sArgs["RotNameSpace"];
  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") 
    << "DDMuonAngular debug: Parent " << parent().name()
    << "\tChild " << childName << "\tNameSpace "
    << idNameSpace << "\tRotation Namespace " << rotns;
#endif
}

void DDMuonAngular::execute(DDCompactView& cpv) {

  double phi    = startAngle;
  int    copyNo = startCopyNo;

  for (int ii=0; ii<n; ii++) {

    double phitmp = phi;
    if (phitmp >= 2._pi) phitmp -= 2._pi;
    DDRotation rotation;
    std::string rotstr("NULL");

    if (std::abs(phitmp) >= 1.0_deg) {
      rotstr = "R" + formatAsDegrees(phitmp);
      rotation = DDRotation(DDName(rotstr, rotns)); 
      if (!rotation) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("MuonGeom") 
	  << "DDMuonAngular: Creating a new rotation "
	  << DDName(rotstr, idNameSpace) << "\t90, " 
	  << convertRadToDeg( phitmp ) << ", 90, "
	  << convertRadToDeg( phitmp + 90._deg ) << ", 0, 0";
#endif
        rotation = DDrot(DDName(rotstr, rotns), 90._deg, phitmp, 90._deg, 90._deg + phitmp, 0., 0.);
      } 
    }
    
    DDTranslation tran(0, 0, zoffset);
  
    DDName parentName = parent().name(); 
    cpv.position(DDName(childName,idNameSpace), parentName, copyNo, tran, rotation);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MuonGeom") 
      << "DDMuonAngular: " << DDName(childName, idNameSpace) << " number " 
      << copyNo << " positioned in " << parentName 
      << " at " << tran << " with " << rotstr << " " << rotation;
#endif
    phi    += stepAngle;
    copyNo += incrCopyNo;
  }
}
