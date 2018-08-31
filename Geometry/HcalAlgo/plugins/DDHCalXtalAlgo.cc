///////////////////////////////////////////////////////////////////////////////
// File: DDHCalXtalAlgo.cc
// Description: Positioning the crystal (layers) in the mother volume
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "Geometry/HcalAlgo/plugins/DDHCalXtalAlgo.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDHCalXtalAlgo::DDHCalXtalAlgo() {
  LogDebug("HCalGeom") << "DDHCalXtalAlgo info: Creating an instance";
}

DDHCalXtalAlgo::~DDHCalXtalAlgo() {}

void DDHCalXtalAlgo::initialize(const DDNumericArguments & nArgs,
				const DDVectorArguments & ,
				const DDMapArguments & ,
				const DDStringArguments & sArgs,
				const DDStringVectorArguments & vsArgs) {

  radius     = nArgs["Radius"];
  offset     = nArgs["Offset"];
  dx         = nArgs["Dx"];
  dz         = nArgs["Dz"];
  angwidth   = nArgs["AngWidth"];
  iaxis      = int (nArgs["Axis"]);
  names      = vsArgs["Names"];

  LogDebug("HCalGeom") << "DDHCalXtalAlgo debug: Parameters for positioning:: "
		       << "Axis " << iaxis << "\tRadius " << radius 
		       << "\tOffset " << offset << "\tDx " << dx << "\tDz " 
		       << dz << "\tAngWidth " << angwidth/CLHEP::deg 
		       << "\tNumbers " << names.size();
  for (unsigned int i = 0; i < names.size(); i++)
    LogDebug("HCalGeom") << "\tnames[" << i << "] = " << names[i];

  DDCurrentNamespace ns;
  idNameSpace = *ns;
  idName      = sArgs["ChildName"]; 
  DDName parentName = parent().name(); 
  LogDebug("HCalGeom") << "DDHCalXtalAlgo debug: Parent " << parentName 
		       << "\tChild " << idName << " NameSpace " << idNameSpace;
}

void DDHCalXtalAlgo::execute(DDCompactView& cpv) {

  double theta[3], phi[3], pos[3];
  phi[0] = 0;
  phi[1] = 90*CLHEP::deg;
  theta[1-iaxis] = 90*CLHEP::deg;
  pos[1-iaxis]   = 0;
  int number = (int)(names.size());
  for (int i=0; i<number; i++) {
    double angle = 0.5*angwidth*(2*i+1-number);
    theta[iaxis] = 90*CLHEP::deg + angle;
    if (angle>0) {
      theta[2]   = angle;
      phi[2]     = 90*iaxis*CLHEP::deg;
    } else {
      theta[2]   =-angle;
      phi[2]     = 90*(2-3*iaxis)*CLHEP::deg;
    }
    pos[iaxis]   = angle*(dz+radius);
    pos[2]       = dx*abs(sin(angle)) + offset;
  
    DDRotation rotation;
    string rotstr = names[i];
    DDTranslation tran(pos[0], pos[1], pos[2]);
    DDName parentName = parent().name(); 

    if (abs(angle) > 0.01*CLHEP::deg) {
      LogDebug("HCalGeom") << "DDHCalXtalAlgo test: Creating a new rotation " 
			   << rotstr << "\t" << theta[0]/CLHEP::deg << "," 
			   << phi[0]/CLHEP::deg << "," << theta[1]/CLHEP::deg 
			   << "," << phi[1]/CLHEP::deg << "," 
			   << theta[2]/CLHEP::deg << ","  << phi[2]/CLHEP::deg;
      rotation = DDrot(DDName(rotstr, idNameSpace), theta[0], phi[0], theta[1],
		       phi[1], theta[2], phi[2]);
    }
    cpv.position(DDName(idName, idNameSpace), parentName, i+1, tran, rotation);
    LogDebug("HCalGeom") << "DDHCalXtalAlgo test: "
			 << DDName(idName,idNameSpace) << " number " << i+1 
			 << " positioned in " << parentName << " at " << tran
			 << " with " << rotation;
  }
}
