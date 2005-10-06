#define DEBUG 0
#define COUT if (DEBUG) cout
///////////////////////////////////////////////////////////////////////////////
// File: DDTECPhiAltAlgo.cc
// Description: Position n copies inside and outside Z at alternate phi values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "Geometry/TrackerSimData/interface/DDTECPhiAltAlgo.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTECPhiAltAlgo::DDTECPhiAltAlgo() {
  COUT << "DDTECPhiAltAlgo info: Creating an instance" << endl;
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

  COUT << "DDTECPhiAltAlgo debug: Parameters for positioning--"
		<< "\tStartAngle " << startAngle/deg << "\tIncrAngle " 
		<< incrAngle/deg << "\tRadius " << radius << "\tZ in/out "
		<< zIn << ", " << zOut << "\tCopy Numbers " << number 
		<< " Start/Increment " << startCopyNo << ", " << incrCopyNo 
		<< endl;

  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name();
  COUT << "DDTECPhiAltAlgo debug: Parent " << parentName 
		<< "\tChild " << childName << " NameSpace " << idNameSpace 
		<< endl;
}

void DDTECPhiAltAlgo::execute() {

  if (number > 0) {
    double theta  = 90.*deg;
    int    copyNo = startCopyNo;

    DDName mother = parent().name();
    DDName child(DDSplit(childName).first, DDSplit(childName).second);
    for (int i=0; i<number; i++) {
      double phiz = startAngle + i*incrAngle;
      double phix = phiz + 90.*deg;
      double phideg = phiz/deg;
  
      DDRotation rotation;
      string rotstr = DDSplit(childName).first + dbl_to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
	COUT << "DDTECPhiAltAlgo test: Creating a new rotation " 
		     << rotstr << "\t" << theta/deg << ", " << phix/deg 
		     << ", 0, 0, " << theta/deg << ", " << phiz/deg << endl;
	rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, 0., 0.,
			 theta, phiz);
      }
	
      double xpos = radius*cos(phiz);
      double ypos = radius*sin(phiz);
      double zpos;
      if (i%2 == 0) zpos = zIn;
      else          zpos = zOut;
      DDTranslation tran(xpos, ypos, zpos);
  
      DDpos (child, mother, copyNo, tran, rotation);
      COUT << "DDTECPhiAltAlgo test: " << child << " number " 
		   << copyNo << " positioned in " << mother << " at " << tran 
		   << " with " << rotation << endl;
      copyNo += incrCopyNo;
    }
  }
}
