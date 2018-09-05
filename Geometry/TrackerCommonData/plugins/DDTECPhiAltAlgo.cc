///////////////////////////////////////////////////////////////////////////////
// File: DDTECPhiAltAlgo.cc
// Description: Position n copies inside and outside Z at alternate phi values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTECPhiAltAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDTECPhiAltAlgo::DDTECPhiAltAlgo() {
  LogDebug("TECGeom") << "DDTECPhiAltAlgo info: Creating an instance";
}

DDTECPhiAltAlgo::~DDTECPhiAltAlgo() {}

void DDTECPhiAltAlgo::initialize(const DDNumericArguments & nArgs,
				 const DDVectorArguments & ,
				 const DDMapArguments & ,
				 const DDStringArguments & sArgs,
				 const DDStringVectorArguments & ) {

  startAngle = nArgs["StartAngle"];
  incrAngle  = nArgs["IncrAngle"];
  radius     = nArgs["Radius"];
  zIn        = nArgs["ZIn"];
  zOut       = nArgs["ZOut"];
  number     = int (nArgs["Number"]);
  startCopyNo= int (nArgs["StartCopyNo"]);
  incrCopyNo = int (nArgs["IncrCopyNo"]);

  LogDebug("TECGeom") << "DDTECPhiAltAlgo debug: Parameters for "
		      << "positioning--" << "\tStartAngle " 
		      << startAngle/CLHEP::deg << "\tIncrAngle " 
		      << incrAngle/CLHEP::deg << "\tRadius " << radius 
		      << "\tZ in/out " << zIn << ", " << zOut 
		      << "\tCopy Numbers " << number  << " Start/Increment " 
		      << startCopyNo << ", " << incrCopyNo;
  DDCurrentNamespace ns;
  idNameSpace = *ns;
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name();
  LogDebug("TECGeom") << "DDTECPhiAltAlgo debug: Parent " << parentName 
		      << "\tChild " << childName << " NameSpace " 
		      << idNameSpace;
}

void DDTECPhiAltAlgo::execute(DDCompactView& cpv) {

  if (number > 0) {
    double theta  = 90.*CLHEP::deg;
    int    copyNo = startCopyNo;

    DDName mother = parent().name();
    DDName child(DDSplit(childName).first, DDSplit(childName).second);
    for (int i=0; i<number; i++) {
      double phiz = startAngle + i*incrAngle;
      double phix = phiz + 90.*CLHEP::deg;
      double phideg = phiz/CLHEP::deg;
  
      DDRotation rotation;
      std::string rotstr = DDSplit(childName).first + std::to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
	LogDebug("TECGeom") << "DDTECPhiAltAlgo test: Creating a new "
			    << "rotation " << rotstr << "\t" 
			    << theta/CLHEP::deg << ", " << phix/CLHEP::deg 
			    << ", 0, 0, " << theta/CLHEP::deg << ", " 
			    << phiz/CLHEP::deg;
	rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, 0., 0.,
			 theta, phiz);
      }
	
      double xpos = radius*cos(phiz);
      double ypos = radius*sin(phiz);
      double zpos;
      if (i%2 == 0) zpos = zIn;
      else          zpos = zOut;
      DDTranslation tran(xpos, ypos, zpos);
  
     cpv.position(child, mother, copyNo, tran, rotation);
      LogDebug("TECGeom") << "DDTECPhiAltAlgo test: " << child <<" number "
			  << copyNo << " positioned in " << mother <<" at "
			  << tran << " with " << rotation;
      copyNo += incrCopyNo;
    }
  }
}
