///////////////////////////////////////////////////////////////////////////////
// File: DDMuonAngular.cc
// Description: Position inside the mother according to (eta,phi) 
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDUnits.h"
#include "Geometry/MuonCommonData/plugins/DDMuonAngular.h"

using namespace dd;
using namespace dd::operators;

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
			   << n << " copies in steps of " << CONVERT_TO( stepAngle, deg )
			   << " from " << CONVERT_TO( startAngle, deg ) << " \tZoffest " 
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

    double phitmp = phi;
    if (phitmp >= 2._pi) phitmp -= 2._pi;
    DDRotation rotation;
    std::string rotstr("NULL");

    if (std::abs(phitmp) >= 1.0_deg) {
      rotstr = "R"; 
      rotstr  += formatAsDegrees(phitmp);
      rotation = DDRotation(DDName(rotstr, rotns)); 
      if (!rotation) {
#ifdef EDM_ML_DEBUG
        edm::LogInfo("MuonGeom") << "DDMuonAngular test: Creating a new rotation "
				 << DDName(rotstr, idNameSpace) << "\t90, " 
				 << CONVERT_TO( phitmp, deg ) << ", 90, " << CONVERT_TO(phitmp + 90._deg, deg) << ", 0, 0";
#endif
        rotation = DDrot(DDName(rotstr, rotns), 90._deg, phitmp, 90._deg, 90._deg + phitmp, 0., 0.);
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
