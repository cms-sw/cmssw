///////////////////////////////////////////////////////////////////////////////
// File: DDTECAxialCableAlgo.cc
// Description: Position n copies at prescribed phi values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/interface/DDTECAxialCableAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

DDTECAxialCableAlgo::DDTECAxialCableAlgo() {
  DCOUT('a', "DDTECAxialCableAlgo info: Creating an instance");
}

DDTECAxialCableAlgo::~DDTECAxialCableAlgo() {}

void DDTECAxialCableAlgo::initialize(const DDNumericArguments & nArgs,
				     const DDVectorArguments & vArgs,
				     const DDMapArguments & ,
				     const DDStringArguments & sArgs,
				     const DDStringVectorArguments & ) {

  n           = int(nArgs["N"]);
  rangeAngle  = nArgs["RangeAngle"];
  zStart      = nArgs["ZStart"];
  zEnd        = nArgs["ZEnd"];
  rMin        = nArgs["RMin"];
  rMax        = nArgs["RMax"];
  width       = nArgs["Width"];
  thickR      = nArgs["ThickR"];
  thickZ      = nArgs["ThickZ"];
  dZ          = nArgs["DZ"];
  startAngle  = vArgs["StartAngle"];
  zPos        = vArgs["ZPos"];
  
  if (fabs(rangeAngle-360.0*deg)<0.001*deg) { 
    delta    =   rangeAngle/double(n);
  } else {
    if (n > 1) {
      delta  =   rangeAngle/double(n-1);
    } else {
      delta  = 0.;
    }
  }  

  DCOUT('A', "DDTECAxialCableAlgo debug: Parameters for creating " << startAngle.size() << " axial cables and positioning " << n << " copies in Service volume");
  DCOUT('A', "                            zStart " << zStart << " zEnd " << zEnd << " rMin " << rMin << " rMax " << rMax << " Cable width " << width/deg << " thickness " << thickR << ", " << thickZ << " dZ " << dZ);
  DCOUT('A', "                            Range, Delta " << rangeAngle/deg << ", " << delta/deg);
  for (unsigned int i=0; i<startAngle.size(); i++)
    DCOUT('A', "                          Cable " << i << " from Z " << zPos[i] << " startAngle " << startAngle[i]/deg);
  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  matName     = sArgs["Material"]; 

  DDName parentName = parent().name();

  DCOUT('A', "DDTECAxialCableAlgo debug: Parent " << parentName << "\tChild " << childName << " NameSpace " << idNameSpace << "\tMaterial " << matName);
}

void DDTECAxialCableAlgo::execute() {

  DDName mother = parent().name();
  double theta  = 90.*deg;

  for (unsigned int k=0; k<startAngle.size(); k++) {

    int i;
    double zv = zPos[k]-dZ-0.5*(zStart+zEnd);
    vector<double> pconZ, pconRmin, pconRmax;
    pconZ.push_back(zv);
    pconRmin.push_back(rMin);
    pconRmax.push_back(rMax);
    pconZ.push_back(zv+thickZ);
    pconRmin.push_back(rMin);
    pconRmax.push_back(rMax);
    pconZ.push_back(zv+thickZ);
    pconRmin.push_back(rMax-thickR);
    pconRmax.push_back(rMax);
    zv = zPos[k]+dZ-0.5*(zStart+zEnd);
    pconZ.push_back(zv-thickZ);
    pconRmin.push_back(rMax-thickR);
    pconRmax.push_back(rMax);
    pconZ.push_back(zv-thickZ);
    pconRmin.push_back(rMin);
    pconRmax.push_back(rMax);
    pconZ.push_back(zv);
    pconRmin.push_back(rMin);
    pconRmax.push_back(rMax);
    pconZ.push_back(zv);
    pconRmin.push_back(rMax-thickR);
    pconRmax.push_back(rMax);
    pconZ.push_back(0.5*(zEnd-zStart));
    pconRmin.push_back(rMax-thickR);
    pconRmax.push_back(rMax);

    string name = childName + dbl_to_string(k);
    DDSolid solid = DDSolidFactory::polycone(DDName(name, idNameSpace),
					     -0.5*width, width, pconZ, 
					     pconRmin, pconRmax);

    DCOUT('a', "DDTECAxialCableAlgo test: " << DDName(name, idNameSpace) << " Polycone made of " << matName << " from " << -0.5*width/deg << " to " << 0.5*width/deg << " and with " << pconZ.size() << " sections ");
    for (unsigned int ii = 0; ii <pconZ.size(); ii++) 
      DCOUT('a', "\t" << "\tZ = " << pconZ[ii] << "\tRmin = "<< pconRmin[ii] << "\tRmax = " << pconRmax[ii]);
    DDName mat(DDSplit(matName).first, DDSplit(matName).second); 
    DDMaterial matter(mat);
    DDLogicalPart genlogic(DDName(name, idNameSpace), matter, solid);
    
    double phi = startAngle[k];
    for (i=0; i<n; i++) {
      double phix = phi;
      double phiy = phix + 90.*deg;
      double phideg = phix/deg;

      DDRotation rotation;
      if (phideg != 0) {
	string rotstr = childName + dbl_to_string(phideg*10.);
	rotation = DDRotation(DDName(rotstr, idNameSpace));
	if (!rotation) {
	  DCOUT('a', "DDTECAxialCableAlgo test: Creating a new rotation: " << rotstr << "\t90., " << phix/deg << ", 90.," << phiy/deg << ", 0, 0");
	  rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, 
			   phiy, 0., 0.);
	}
      }
	
      DDTranslation tran(0,0,0);
      DDpos (DDName(name, idNameSpace), mother, i+1, tran, rotation);
      DCOUT('a', "DDTECAxialCableAlgo test " << DDName(name, idNameSpace) << " number " << i+1 << " positioned in " << mother << " at " << tran  << " with "  << rotation);

      phi  += delta;
    }
  }
}
