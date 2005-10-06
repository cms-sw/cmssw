#define DEBUG 0
#define COUT if (DEBUG) cout
///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerPhiAlgo.cc
// Description: Position n copies at prescribed phi values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "Geometry/TrackerSimData/interface/DDTrackerPhiAlgo.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTrackerPhiAlgo::DDTrackerPhiAlgo() {
  COUT << "DDTrackerPhiAlgo info: Creating an instance" << endl;
}

DDTrackerPhiAlgo::~DDTrackerPhiAlgo() {}

void DDTrackerPhiAlgo::initialize(const DDNumericArguments & nArgs,
				  const DDVectorArguments & vArgs,
				  const DDMapArguments & ,
				  const DDStringArguments & sArgs,
				  const DDStringVectorArguments & ) {

  radius     = nArgs["Radius"];
  tilt       = nArgs["Tilt"];
  phi        = vArgs["Phi"];
  zpos       = vArgs["ZPos"];

  COUT << "DDTrackerPhiAlgo debug: Parameters for positioning:: "
		<< " Radius " << radius << " Tilt " << tilt/deg << " Copies " 
		<< phi.size() << " at phi/z values:" << endl;
  for (unsigned int i=0; i<phi.size(); i++)
    COUT << " " << phi[i]/deg << " " << zpos[i];
  COUT << endl;

  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name();
  COUT << "DDTrackerPhiAlgo debug: Parent " << parentName <<"\tChild "
		<< childName << " NameSpace " << idNameSpace << endl;
}

void DDTrackerPhiAlgo::execute() {

  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  double theta  = 90.*deg;
  for (unsigned int i=0; i<phi.size(); i++) {
    double phix = phi[i] + tilt;
    double phiy = phix + 90.*deg;
    double phideg = phi[i]/deg;

    string rotstr = DDSplit(childName).first + dbl_to_string(phideg);
    DDRotation rotation = DDRotation(DDName(rotstr, idNameSpace));
    if (!rotation) {
      COUT << "DDTrackerPhiAlgo test: Creating a new rotation: " 
		   << rotstr << "\t" << "90., " << phix/deg << ", 90.," 
		   << phiy/deg << ", 0, 0" << endl;
      rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
		       0., 0.);
    }
	
    double xpos = radius*cos(phi[i]);
    double ypos = radius*sin(phi[i]);
    DDTranslation tran(xpos, ypos, zpos[i]);
  
    DDpos (child, mother, i+1, tran, rotation);
    COUT << "DDTrackerPhiAlgo test: " << child << " number " << i+1 
		 << " positioned in " << mother << " at " << tran  << " with " 
		 << rotation << endl;
  }
}
