#define DEBUG 0
#define COUT if (DEBUG) cout
///////////////////////////////////////////////////////////////////////////////
// File: DDTIDAxialCableAlgo.cc
// Description: Create and position TID axial cables at prescribed phi values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "Geometry/TrackerSimData/interface/DDTIDAxialCableAlgo.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTIDAxialCableAlgo::DDTIDAxialCableAlgo() {
  COUT << "DDTIDAxialCableAlgo info: Creating an instance" << endl;
}

DDTIDAxialCableAlgo::~DDTIDAxialCableAlgo() {}

void DDTIDAxialCableAlgo::initialize(const DDNumericArguments & nArgs,
				     const DDVectorArguments & vArgs,
				     const DDMapArguments & ,
				     const DDStringArguments & sArgs,
				     const DDStringVectorArguments & ) {

  zBend       = nArgs["ZBend"];
  zEnd        = nArgs["ZEnd"];
  rMin        = nArgs["RMin"];
  rMax        = nArgs["RMax"];
  rTop        = nArgs["RTop"];
  width       = nArgs["Width"];
  thick       = nArgs["Thick"];
  angles      = vArgs["Angles"];
  zposWheel   = vArgs["ZPosWheel"];
  zposRing    = vArgs["ZPosRing"];

  COUT << "DDTIDAxialCableAlgo debug: Parameters for creating "
		<< (zposWheel.size()+2) << " axial cables and positioning "
		<< angles.size() << " copies in Service volume" << endl
		<< "                            zBend " << zBend << " zEnd " 
		<< zEnd << " rMin " << rMin << " rMax " << rMax 
		<< " Cable width " << width/deg << " thickness " << thick 
		<< endl << "                            Angles";
  for (unsigned int i=0; i<angles.size(); i++)
    COUT << " " << angles[i]/deg;
  COUT << endl << "                          Wheels " 
		<< zposWheel.size() << " at Z";
  for (unsigned int i=0; i<zposWheel.size(); i++)
    COUT << " " << zposWheel[i];
  COUT << endl << "                          each with " 
		<< zposRing.size() << " Rings at Z";
  for (unsigned int i=0; i<zposRing.size(); i++)
    COUT << " " << zposRing[i];
  COUT << endl;

  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  matIn       = sArgs["MaterialIn"]; 
  matOut      = sArgs["MaterialOut"]; 

  DDName parentName = parent().name();
  COUT << "DDTIDAxialCableAlgo debug: Parent " << parentName 
		<< "\tChild " << childName << " NameSpace " << idNameSpace 
		<< "\tMaterial " << matIn << " and " << matOut << endl;
}

void DDTIDAxialCableAlgo::execute() {

  DDName mother = parent().name();
  vector<DDName> logs;
  double thk = thick/zposRing.size();
  double r   = rMin;
  double thktot = 0;
  double z;

  //Cables between the wheels
  for (unsigned int k=0; k<zposWheel.size(); k++) {

    vector<double> pconZ, pconRmin, pconRmax;
    for (unsigned int i=0; i<zposRing.size(); i++) {
      thktot += thk;
      z       = zposWheel[k] + zposRing[i] - 0.5*thk;
      if (i != 0) {
	pconZ.push_back(z);
	pconRmin.push_back(r);
	pconRmax.push_back(rMax);
      }
      r       = rMin;
      pconZ.push_back(z);
      pconRmin.push_back(r);
      pconRmax.push_back(rMax);
      z      += thk;
      pconZ.push_back(z);
      pconRmin.push_back(r);
      pconRmax.push_back(rMax);
      r       = rMax - thktot;
      pconZ.push_back(z);
      pconRmin.push_back(r);
      pconRmax.push_back(rMax);
    }
    if (k >= zposWheel.size()-1) z = zBend;
    else                         z = zposWheel[k+1] + zposRing[0] - 0.5*thk;
    pconZ.push_back(z);
    pconRmin.push_back(r);
    pconRmax.push_back(rMax);
    
    string name = childName + dbl_to_string(k);
    DDSolid solid = DDSolidFactory::polycone(DDName(name, idNameSpace),
					     -0.5*width, width, pconZ, 
					     pconRmin, pconRmax);

    COUT << "DDTIDAxialCableAlgo test: " << DDName(name, idNameSpace)
		 << " Polycone made of " << matIn << " from " 
		 << -0.5*width/deg << " to " << 0.5*width/deg << " and with " 
		 << pconZ.size() << " sections "  << endl;
    for (unsigned int i = 0; i <pconZ.size(); i++) 
      COUT << "\t" << "\tZ = " << pconZ[i] << "\tRmin = "<< pconRmin[i]
		   << "\tRmax = " << pconRmax[i] << endl;

    DDName mat(DDSplit(matIn).first, DDSplit(matIn).second); 
    DDMaterial matter(mat);
    DDLogicalPart genlogic(DDName(name, idNameSpace), matter, solid);
    logs.push_back(DDName(name, idNameSpace));
  }

  //Cable in the vertical part
  vector<double> pconZ, pconRmin, pconRmax;
  r = thktot*rMax/rTop;
  z = zBend - thktot;
  COUT << "DDTIDAxialCableAlgo test: Thk " << thk << " Total "
	       << thktot << " rMax " << rMax << " rTop " << rTop << " dR "
	       << r << " z " << z << endl;
  pconZ.push_back(z);
  pconRmin.push_back(rMax);
  pconRmax.push_back(rMax);
  z = zBend - r;
  pconZ.push_back(z);
  pconRmin.push_back(rMax);
  pconRmax.push_back(rTop);
  pconZ.push_back(zBend);
  pconRmin.push_back(rMax);
  pconRmax.push_back(rTop);

  string name = childName + dbl_to_string(zposWheel.size());
  DDSolid solid = DDSolidFactory::polycone(DDName(name, idNameSpace),
					   -0.5*width, width, pconZ, 
					   pconRmin, pconRmax);

  COUT << "DDTIDAxialCableAlgo test: " << DDName(name, idNameSpace)
	       << " Polycone made of " << matIn << " from " 
	       << -0.5*width/deg << " to " << 0.5*width/deg << " and with " 
	       << pconZ.size() << " sections "  << endl;
  for (unsigned int i = 0; i <pconZ.size(); i++) 
    COUT << "\t" << "\tZ = " << pconZ[i] << "\tRmin = "<< pconRmin[i]
		 << "\tRmax = " << pconRmax[i] << endl;

  DDName mat(DDSplit(matIn).first, DDSplit(matIn).second); 
  DDMaterial matter(mat);
  DDLogicalPart genlogic(DDName(name, idNameSpace), matter, solid);
  logs.push_back(DDName(name, idNameSpace));

  //Cable in the outer part
  name = childName + dbl_to_string(zposWheel.size()+1);
  r    = rTop-r;
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*(zEnd-zBend),
                               r, rTop, -0.5*width, width);
  COUT << "DDTIDAxialCableAlgo test: " << DDName(name, idNameSpace)
               << " Tubs made of " << matOut << " from " << -0.5*width/deg 
	       << " to " << 0.5*width/deg << " with Rin " << r << " Rout " 
	       << rTop << " ZHalf " << 0.5*(zEnd-zBend) << endl;
  mat    = DDName(DDSplit(matOut).first, DDSplit(matOut).second);
  matter = DDMaterial(mat);
  genlogic = DDLogicalPart(DDName(name, idNameSpace), matter, solid);
  logs.push_back(DDName(name, idNameSpace));

  //Position the cables
  double theta = 90.*deg;
  for (unsigned int i=0; i<angles.size(); i++) {
    double phix = angles[i];
    double phiy = phix + 90.*deg;
    double phideg = phix/deg;

    DDRotation rotation;
    if (phideg != 0) {
      string rotstr = childName + dbl_to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
	COUT << "DDTIDAxialCableAlgo test: Creating a new rotation: "
		     << rotstr << " " << theta/deg << ", " << phix/deg << ", " 
		     << theta/deg << ", " << phiy/deg << ", 0, 0" << endl;
	rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, 
			 phiy, 0., 0.);
      }
    }
    
    for (unsigned int k=0; k<logs.size(); k++) {
      DDTranslation tran(0,0,0);
      if (k == logs.size()-1) tran = DDTranslation(0,0,0.5*(zEnd+zBend));
      DDpos (logs[k], mother, i+1, tran, rotation);
      COUT << "DDTIDAxialCableAlgo test " << logs[k]
		   << " number " << i+1 << " positioned in " << mother 
		   << " at " << tran  << " with "  << rotation << endl;
    }
  }
}
