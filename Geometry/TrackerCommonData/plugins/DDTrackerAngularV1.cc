///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerAngularV1.cc
// Description: Position n copies at prescribed phi values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTrackerAngularV1.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDTrackerAngularV1::DDTrackerAngularV1() {
  LogDebug("TrackerGeom") << "DDTrackerAngularV1 info: Creating an instance";
}

DDTrackerAngularV1::~DDTrackerAngularV1() {}

void DDTrackerAngularV1::initialize(const DDNumericArguments & nArgs,
				  const DDVectorArguments & vArgs,
				  const DDMapArguments & ,
				  const DDStringArguments & sArgs,
				  const DDStringVectorArguments & ) {

  n           = int(nArgs["N"]);
  startCopyNo = int(nArgs["StartCopyNo"]);
  incrCopyNo  = int(nArgs["IncrCopyNo"]);
  rangeAngle  = nArgs["RangeAngle"];
  startAngle  = nArgs["StartAngle"];
  radius      = nArgs["Radius"];
  center      = vArgs["Center"];
  
  if (fabs(rangeAngle-360.0*CLHEP::deg)<0.001*CLHEP::deg) { 
    delta    =   rangeAngle/double(n);
  } else {
    if (n > 1) {
      delta    =   rangeAngle/double(n-1);
    } else {
      delta = 0.;
    }
  }  

  LogDebug("TrackerGeom") << "DDTrackerAngularV1 debug: Parameters for position"
			  << "ing:: n " << n << " Start, Range, Delta " 
			  << startAngle/CLHEP::deg << " " 
			  << rangeAngle/CLHEP::deg << " " << delta/CLHEP::deg
			  << " Radius " << radius << " Centre " << center[0] 
			  << ", " << center[1] << ", "<<center[2];
  DDCurrentNamespace ns;
  idNameSpace = *ns;
  childName   = sArgs["ChildName"]; 

  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDTrackerAngularV1 debug: Parent " << parentName 
			  << "\tChild " << childName << " NameSpace "
			  << idNameSpace;
}

void DDTrackerAngularV1::execute(DDCompactView& cpv) {

  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  double theta  = 90.*CLHEP::deg;
  int    copy   = startCopyNo;
  double phi    = startAngle;
  for (int i=0; i<n; i++) {
    double phix = phi + 90.*CLHEP::deg;
    double phiy = phix + 90.*CLHEP::deg;
    double phideg = phix/CLHEP::deg;

    DDRotation rotation;
    if (phideg != 0) {
      std::string rotstr = DDSplit(childName).first + std::to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
	LogDebug("TrackerGeom") << "DDTrackerAngularV1 test: Creating a new "
				<< "rotation: " << rotstr << "\t90., " 
				<< phix/CLHEP::deg << ", 90.," 
				<< phiy/CLHEP::deg <<", 0, 0";
	rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
			 0., 0.);
      }
    }
	
    double xpos = radius*cos(phi) + center[0];
    double ypos = radius*sin(phi) + center[1];
    double zpos = center[2];
    DDTranslation tran(xpos, ypos, zpos);
  
   cpv.position(child, mother, copy, tran, rotation);
    LogDebug("TrackerGeom") << "DDTrackerAngularV1 test " << child << " number " 
			    << copy << " positioned in " << mother << " at "
			    << tran  << " with " << rotation;
    copy += incrCopyNo;
    phi  += delta;
  }
}
