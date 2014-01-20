///////////////////////////////////////////////////////////////////////////////
// File: DDBHMAngular.cc
// Description: Position inside the mother according to phi
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "DDBHMAngular.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDBHMAngular::DDBHMAngular() {
  LogDebug("HCalGeom") << "DDBHMAngular test: Creating an instance";
}

DDBHMAngular::~DDBHMAngular() {}

void DDBHMAngular::initialize(const DDNumericArguments & nArgs,
			      const DDVectorArguments & ,
			      const DDMapArguments & ,
			      const DDStringArguments & sArgs,
			      const DDStringVectorArguments & ) {

  units  = int (nArgs["number"]);
  rr     = nArgs["radius"];
  dphi   = nArgs["deltaPhi"];
  LogDebug("HCalGeom") << "DDBHMAngular debug: Parameters for positioning-- "
		       << units << " copies at radius " << rr/CLHEP::cm 
		       << " cm with delta(phi) " << dphi/CLHEP::deg;

  rotMat      = sArgs["Rotation"];
  childName   = sArgs["ChildName"]; 
  LogDebug("HCalGeom") << "DDBHMAngular debug: Parent " << parent().name()
		       << "\tChild " << childName << "\tRotation matrix " 
		       << rotMat;
}

void DDBHMAngular::execute(DDCompactView& cpv) {

  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  DDName parentName  = parent().name(); 
  std::string rotstr = DDSplit(rotMat).first;
  DDRotation rot;
  if (rotstr != "NULL") {
    std::string rotns  = DDSplit(rotMat).second;
    rot = DDRotation(DDName(rotstr, rotns));
  }

  for (int jj=0; jj<units; jj++) {
    double driverX(0), driverY(0), driverZ(0);
    if (jj<16){                             
      driverX = rr*cos((jj+0.5)*dphi);
      driverY = sqrt(rr*rr-driverX*driverX);
    } else if (jj==16) { 
      driverX = rr*cos(15.5*dphi);
      driverY =-sqrt(rr*rr-driverX*driverX);                      
    } else if (jj==17) { 
      driverX = rr*cos(14.5*dphi);
      driverY =-sqrt(rr*rr-driverX*driverX);                      
    } else if (jj==18) { 
      driverX = rr*cos(0.5*dphi);
      driverY =-sqrt(rr*rr-driverX*driverX);                      
    } else if (jj==19) { 
      driverX = rr*cos(1.5*dphi);
      driverY =-sqrt(rr*rr-driverX*driverX); 
    }               
    DDTranslation tran(driverX, driverY, driverZ);
  
    cpv.position(child, parentName, jj+1, tran, rot);
    LogDebug("HCalGeom") << "DDBHMAngular test: " << child << " number " <<jj+1
			 << " positioned in " << parentName << " at " << tran 
			 << " with " << rot;
  }
}
