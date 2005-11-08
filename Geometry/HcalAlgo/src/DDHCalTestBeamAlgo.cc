///////////////////////////////////////////////////////////////////////////////
// File: DDHCalTestBeamAlgo.cc
// Description: Position inside the mother according to (eta,phi) 
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HcalAlgo/interface/DDHCalTestBeamAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

DDHCalTestBeamAlgo::DDHCalTestBeamAlgo() {
  DCOUT('a', "DDHCalTestBeamAlgo test: Creating an instance");
}

DDHCalTestBeamAlgo::~DDHCalTestBeamAlgo() {}

void DDHCalTestBeamAlgo::initialize(const DDNumericArguments & nArgs,
				    const DDVectorArguments & ,
				    const DDMapArguments & ,
				    const DDStringArguments & sArgs,
				    const DDStringVectorArguments & ) {

  eta        = nArgs["Eta"];
  theta      = 2.0*atan(exp(-eta));
  phi        = nArgs["Phi"];
  distance   = nArgs["Dist"];
  distanceZ  = nArgs["DistZ"];
  dz         = nArgs["Dz"];
  copyNumber = int (nArgs["Number"]);
  dist       = (distance+distanceZ/sin(theta));
  DCOUT('A', "DDHCalTestBeamAlgo debug: Parameters for positioning--" << " Eta " << eta << "\tPhi " << phi/deg << "\tTheta " << theta/deg << "\tDistance " << distance << "/" << distanceZ << "/" << dist << "\tDz " << dz << "\tcopyNumber " << copyNumber);

  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name(); 
  DCOUT('A', "DDHCalTestBeamAlgo debug: Parent " << parentName << "\tChild " << childName << " NameSpace " << idNameSpace);
}

void DDHCalTestBeamAlgo::execute() {

  double thetax = 90.*deg + theta;
  double sthx   = sin(thetax);
  if (abs(sthx)>1.e-12) sthx = 1./sthx;
  else                  sthx = 1.;
  double phix   = atan2(sthx*cos(theta)*sin(phi),sthx*cos(theta)*cos(phi));
  double thetay = 90.*deg;
  double phiy   = 90.*deg + phi;
  double thetaz = theta;
  double phiz   = phi;
  
  DDRotation rotation;
  string rotstr = childName;
  DCOUT('a', "DDHCalTestBeamAlgo test: Creating a new rotation " << rotstr << "\t" << thetax/deg << "," << phix/deg << "," << thetay/deg << "," << phiy/deg <<"," << thetaz/deg <<"," << phiz/deg);
  rotation = DDrot(DDName(rotstr, idNameSpace), thetax, phix, thetay, phiy,
		   thetaz, phiz);
	
  double r    = dist*sin(theta);
  double xpos = r*cos(phi);
  double ypos = r*sin(phi);
  double zpos = dist*cos(theta);
  DDTranslation tran(xpos, ypos, zpos);
  
  DDName parentName = parent().name(); 
  DDpos (DDName(childName,idNameSpace), parentName,copyNumber, tran,rotation);
  DCOUT('a', "DDHCalTestBeamAlgo test: " << DDName(childName, idNameSpace) << " number " << copyNumber << " positioned in " << parentName << " at " << tran << " with " << rotation);

  xpos = (dist-dz)*sin(theta)*cos(phi);
  ypos = (dist-dz)*sin(theta)*sin(phi);
  zpos = (dist-dz)*cos(theta);

  std::cout << "DDHCalTestBeamAlgo info: Suggested Beam position (" << xpos 
	    << ", " << ypos << ", " << zpos << ") and (eta, phi) = (" << eta
	    << ", " << phi << ")" << std::endl;
}
