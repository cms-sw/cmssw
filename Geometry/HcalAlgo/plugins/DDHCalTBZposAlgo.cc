///////////////////////////////////////////////////////////////////////////////
// File: DDHCalTBZposAlgo.cc
// Description: Position inside the mother by shifting along Z given by eta
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HcalAlgo/plugins/DDHCalTBZposAlgo.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDHCalTBZposAlgo::DDHCalTBZposAlgo() {
  LogDebug("HCalGeom") << "DDHCalTBZposAlgo test: Creating an instance";
}

DDHCalTBZposAlgo::~DDHCalTBZposAlgo() {}

void DDHCalTBZposAlgo::initialize(const DDNumericArguments & nArgs,
				  const DDVectorArguments & ,
				  const DDMapArguments & ,
				  const DDStringArguments & sArgs,
				  const DDStringVectorArguments & ) {

  eta        = nArgs["Eta"];
  theta      = 2.0*atan(exp(-eta));
  shiftX     = nArgs["ShiftX"];
  shiftY     = nArgs["ShiftY"];
  zoffset    = nArgs["Zoffset"];
  dist       = nArgs["Distance"];
  tilt       = nArgs["TiltAngle"];
  copyNumber = int (nArgs["Number"]);
  LogDebug("HCalGeom") << "DDHCalTBZposAlgo debug: Parameters for position"
		       << "ing--" << " Eta " << eta << "\tTheta " 
		       << theta/CLHEP::deg << "\tShifts " << shiftX << ", " 
		       << shiftY  << " along x, y axes; \tZoffest " << zoffset
		       << "\tRadial Distance " << dist << "\tTilt angle "
		       << tilt/CLHEP::deg << "\tcopyNumber " << copyNumber;

  DDCurrentNamespace ns;
  idNameSpace = *ns;
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name(); 
  LogDebug("HCalGeom") << "DDHCalTBZposAlgo debug: Parent " << parentName
		       << "\tChild " << childName << " NameSpace "
		       << idNameSpace;
}

void DDHCalTBZposAlgo::execute(DDCompactView& cpv) {

  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
 
  double thetax = 90.*CLHEP::deg - theta;
  double z      = zoffset + dist*tan(thetax);
  double x      = shiftX - shiftY*sin(tilt);
  double y      = shiftY*cos(tilt);
  DDTranslation tran(x,y,z);
  DDRotation rot;
  double tiltdeg = tilt/CLHEP::deg;
  int    itilt   = int(tiltdeg+0.1);
  if (itilt != 0) {
    std::string rotstr = "R";
    if (tiltdeg < 100) rotstr = "R0"; 
    rotstr = rotstr + std::to_string(tiltdeg);
    rot    = DDRotation(DDName(rotstr, idNameSpace)); 
    if (!rot) {
      LogDebug("HCalGeom") << "DDHCalAngular test: Creating a new rotation "
			   << DDName(rotstr,idNameSpace) << "\t90, " << tiltdeg
			   << ", 90, " << (tiltdeg+90) << ", 0, 0";
      rot = DDrot(DDName(rotstr, idNameSpace), 90*CLHEP::deg, tilt, 
		  90*CLHEP::deg, (90*CLHEP::deg+tilt), 0.0,  0.0);
    }
  }
 cpv.position(child, mother, copyNumber, tran, rot);
  LogDebug("HCalGeom") << "DDHCalTBZposAlgo test: " << child << " number " 
		       << copyNumber << " positioned in " << mother
		       << " at " << tran << " with " << rot;
}
