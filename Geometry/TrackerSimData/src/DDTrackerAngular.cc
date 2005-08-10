#define DEBUG 0
#define COUT if (DEBUG) cout
///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerAngular.cc
// Description: Position n copies at prescribed phi values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "Geometry/TrackerSimData/interface/DDTrackerAngular.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTrackerAngular::DDTrackerAngular() {
  COUT << "DDTrackerAngular info: Creating an instance" << endl;
}

DDTrackerAngular::~DDTrackerAngular() {}

void DDTrackerAngular::initialize(const DDNumericArguments & nArgs,
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
  
  if (fabs(rangeAngle-360.0*deg)<0.001*deg) { 
    delta    =   rangeAngle/double(n);
  } else {
    if (n > 1) {
      delta    =   rangeAngle/double(n-1);
    } else {
      delta = 0.;
    }
  }  

  COUT << "DDTrackerAngular debug: Parameters for positioning:: n " 
		<< n << " Start, Range, Delta " << startAngle/deg << " " 
		<< rangeAngle/deg << " " << delta/deg << " Radius " << radius 
		<< " Centre:";
  for (unsigned int i=0; i<center.size(); i++)
    COUT << " " << center[i];
  COUT << endl;

  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 

  DDName parentName = parent().name();
  COUT << "DDTrackerAngular debug: Parent " << parentName 
		<< "\tChild " << childName << " NameSpace " << idNameSpace 
		<< endl;
}

void DDTrackerAngular::execute() {

  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  double theta  = 90.*deg;
  int    copy   = startCopyNo;
  double phi    = startAngle;
  for (int i=0; i<n; i++) {
    double phix = phi;
    double phiy = phix + 90.*deg;
    double phideg = phix/deg;

    DDRotation rotation;
    if (phideg != 0) {
      string rotstr = DDSplit(childName).first + dbl_to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
	COUT << "DDTrackerAngular test: Creating a new rotation: " 
		     << rotstr << "\t90., " << phix/deg << ", 90.," << phiy/deg
		     << ", 0, 0" << endl;
	rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
			 0., 0.);
      }
    }
	
    double xpos = radius*cos(phi) + center[0];
    double ypos = radius*sin(phi) + center[1];
    double zpos = center[2];
    DDTranslation tran(xpos, ypos, zpos);
  
    DDpos (child, mother, copy, tran, rotation);
    COUT << "DDTrackerAngular test " << child << " number " << copy 
		 << " positioned in " << mother << " at " << tran  << " with " 
		 << rotation << endl;
    copy += incrCopyNo;
    phi  += delta;
  }
}
