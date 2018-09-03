///////////////////////////////////////////////////////////////////////////////
// File: DDTECPhiAlgo.cc
// Description: Position n copies inside and outside Z at alternate phi values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTECPhiAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDTECPhiAlgo::DDTECPhiAlgo() {
  LogDebug("TECGeom") << "DDTECPhiAlgo info: Creating an instance";
}

DDTECPhiAlgo::~DDTECPhiAlgo() {}

void DDTECPhiAlgo::initialize(const DDNumericArguments & nArgs,
			      const DDVectorArguments & ,
			      const DDMapArguments & ,
			      const DDStringArguments & sArgs,
			      const DDStringVectorArguments & ) {

  startAngle = nArgs["StartAngle"];
  incrAngle  = nArgs["IncrAngle"];
  zIn        = nArgs["ZIn"];
  zOut       = nArgs["ZOut"];
  number     = int (nArgs["Number"]);
  startCopyNo= int (nArgs["StartCopyNo"]);
  incrCopyNo = int (nArgs["IncrCopyNo"]);

  LogDebug("TECGeom") << "DDTECPhiAlgo debug: Parameters for "
		      << "positioning--" << "\tStartAngle " 
		      << startAngle/CLHEP::deg << "\tIncrAngle " 
		      << incrAngle/CLHEP::deg << "\tZ in/out " << zIn << ", " 
		      << zOut 	      << "\tCopy Numbers " << number 
		      << " Start/Increment " << startCopyNo << ", " 
		      << incrCopyNo;
  DDCurrentNamespace ns;
  idNameSpace = *ns;
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name();
  LogDebug("TECGeom") << "DDTECPhiAlgo debug: Parent " << parentName 
		      << "\tChild " << childName << " NameSpace " 
		      << idNameSpace;
}

void DDTECPhiAlgo::execute(DDCompactView& cpv) {

  if (number > 0) {
    double theta  = 90.*CLHEP::deg;
    int    copyNo = startCopyNo;

    DDName mother = parent().name();
    DDName child(DDSplit(childName).first, DDSplit(childName).second);
    for (int i=0; i<number; i++) {
      double phix = startAngle + i*incrAngle;
      double phiy = phix + 90.*CLHEP::deg;
      double phideg = phix/CLHEP::deg;
  
      DDRotation rotation;
      std::string rotstr = DDSplit(childName).first + std::to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
	LogDebug("TECGeom") << "DDTECPhiAlgo test: Creating a new "
			    << "rotation " << rotstr << "\t" 
			    << theta/CLHEP::deg << ", " << phix/CLHEP::deg 
			    << ", " << theta/CLHEP::deg << ", "
			    << phiy/CLHEP::deg << ", 0, 0";
	rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
			 0., 0.);
      }
	
      double zpos = zOut;
      if (i%2 == 0) zpos = zIn;
      DDTranslation tran(0., 0., zpos);
  
     cpv.position(child, mother, copyNo, tran, rotation);
      LogDebug("TECGeom") << "DDTECPhiAlgo test: " << child <<" number "
			  << copyNo << " positioned in " << mother <<" at "
			  << tran << " with " << rotation;
      copyNo += incrCopyNo;
    }
  }
}
