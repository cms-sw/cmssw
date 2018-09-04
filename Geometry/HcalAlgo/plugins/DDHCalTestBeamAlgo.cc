///////////////////////////////////////////////////////////////////////////////
// File: DDHCalTestBeamAlgo.cc
// Description: Position inside the mother according to (eta,phi) 
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "Geometry/HcalAlgo/plugins/DDHCalTestBeamAlgo.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDHCalTestBeamAlgo::DDHCalTestBeamAlgo() {
  LogDebug("HCalGeom") << "DDHCalTestBeamAlgo test: Creating an instance";
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
  LogDebug("HCalGeom") << "DDHCalTestBeamAlgo debug: Parameters for position"
		       << "ing--" << " Eta " << eta << "\tPhi " 
		       << phi/CLHEP::deg << "\tTheta " << theta/CLHEP::deg 
		       << "\tDistance " << distance << "/" << distanceZ << "/"
		       << dist <<"\tDz " << dz <<"\tcopyNumber " << copyNumber;

  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  DDName parentName = parent().name(); 
  LogDebug("HCalGeom") << "DDHCalTestBeamAlgo debug: Parent " << parentName
		       << "\tChild " << childName << " NameSpace "
		       << idNameSpace;
}

void DDHCalTestBeamAlgo::execute(DDCompactView& cpv) {

  double thetax = 90.*CLHEP::deg + theta;
  double sthx   = sin(thetax);
  if (abs(sthx)>1.e-12) sthx = 1./sthx;
  else                  sthx = 1.;
  double phix   = atan2(sthx*cos(theta)*sin(phi),sthx*cos(theta)*cos(phi));
  double thetay = 90.*CLHEP::deg;
  double phiy   = 90.*CLHEP::deg + phi;
  double thetaz = theta;
  double phiz   = phi;
  
  DDRotation rotation;
  string rotstr = childName;
  LogDebug("HCalGeom") << "DDHCalTestBeamAlgo test: Creating a new rotation "
		       << rotstr << "\t" << thetax/CLHEP::deg << "," 
		       << phix/CLHEP::deg << "," << thetay/CLHEP::deg << "," 
		       << phiy/CLHEP::deg << "," << thetaz/CLHEP::deg <<"," 
		       << phiz/CLHEP::deg;
  rotation = DDrot(DDName(rotstr, idNameSpace), thetax, phix, thetay, phiy,
		   thetaz, phiz);
	
  double r    = dist*sin(theta);
  double xpos = r*cos(phi);
  double ypos = r*sin(phi);
  double zpos = dist*cos(theta);
  DDTranslation tran(xpos, ypos, zpos);
  
  DDName parentName = parent().name(); 
 cpv.position(DDName(childName,idNameSpace), parentName,copyNumber, tran,rotation);
  LogDebug("HCalGeom") << "DDHCalTestBeamAlgo test: " 
		       << DDName(childName, idNameSpace) << " number " 
		       << copyNumber << " positioned in " << parentName 
		       << " at " << tran << " with " << rotation;

  xpos = (dist-dz)*sin(theta)*cos(phi);
  ypos = (dist-dz)*sin(theta)*sin(phi);
  zpos = (dist-dz)*cos(theta);

  edm::LogInfo("HCalGeom") << "DDHCalTestBeamAlgo: Suggested Beam position "
			   << "(" << xpos << ", " << ypos << ", " << zpos 
			   << ") and (dist, eta, phi) = (" << (dist-dz) << ", "
			   << eta << ", " << phi << ")";
}
