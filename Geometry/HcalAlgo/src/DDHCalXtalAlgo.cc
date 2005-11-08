///////////////////////////////////////////////////////////////////////////////
// File: DDHCalXtalAlgo.cc
// Description: Positioning the crystal (layers) in the mother volume
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
#include "Geometry/HcalAlgo/interface/DDHCalXtalAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

DDHCalXtalAlgo::DDHCalXtalAlgo() {
  DCOUT('a', "DDHCalXtalAlgo info: Creating an instance");
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

  DCOUT('A', "DDHCalXtalAlgo debug: Parameters for positioning:: Axis " << iaxis << "\tRadius " << radius << "\tOffset " << offset << "\tDx " << dx << "\tDz " << dz << "\tAngWidth " << angwidth/deg << "\tNumbers " << names.size());
  for (unsigned int i = 0; i < names.size(); i++)
    DCOUT('A', "\t" << names[i]);
  //DCOUT('A', );

  idNameSpace = DDCurrentNamespace::ns();
  idName      = sArgs["ChildName"]; 
  DDName parentName = parent().name(); 
  DCOUT('A', "DDHCalXtalAlgo debug: Parent " << parentName << "\tChild " << idName << " NameSpace " << idNameSpace);
}

void DDHCalXtalAlgo::execute() {

  double theta[3], phi[3], pos[3];
  phi[0] = 0;
  phi[1] = 90*deg;
  theta[1-iaxis] = 90*deg;
  pos[1-iaxis]   = 0;
  int number = (int)(names.size());
  for (int i=0; i<number; i++) {
    double angle = 0.5*angwidth*(2*i+1-number);
    theta[iaxis] = 90*deg + angle;
    if (angle>0) {
      theta[2]   = angle;
      phi[2]     = 90*iaxis*deg;
    } else {
      theta[2]   =-angle;
      phi[2]     = 90*(2-3*iaxis)*deg;
    }
    pos[iaxis]   = angle*(dz+radius);
    pos[2]       = dx*abs(sin(angle)) + offset;
  
    DDRotation rotation;
    string rotstr = names[i];
    DDTranslation tran(pos[0], pos[1], pos[2]);
    DDName parentName = parent().name(); 

    if (abs(angle) > 0.01*deg) {
      DCOUT('a', "DDHCalXtalAlgo test: Creating a new rotation " << rotstr << "\t" << theta[0]/deg << "," << phi[0]/deg << "," << theta[1]/deg << "," << phi[1]/deg << "," << theta[2]/deg << "," << phi[2]/deg);
      rotation = DDrot(DDName(rotstr, idNameSpace), theta[0], phi[0], theta[1],
		       phi[1], theta[2], phi[2]);
    }
    DDpos (DDName(idName, idNameSpace), parentName, i+1, tran, rotation);
    DCOUT('a', "DDHCalXtalAlgo test: " << DDName(idName, idNameSpace) << " number " << i+1 << " positioned in " << parentName << " at " << tran << " with " << rotation);
  }
}
