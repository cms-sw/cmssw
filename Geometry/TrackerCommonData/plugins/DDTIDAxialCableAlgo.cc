///////////////////////////////////////////////////////////////////////////////
// File: DDTIDAxialCableAlgo.cc
// Description: Create and position TID axial cables at prescribed phi values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTIDAxialCableAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDTIDAxialCableAlgo::DDTIDAxialCableAlgo() {
  LogDebug("TIDGeom") << "DDTIDAxialCableAlgo info: Creating an instance";
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

  LogDebug("TIDGeom") << "DDTIDAxialCableAlgo debug: Parameters for creating "
		      << (zposWheel.size()+2) << " axial cables and position"
		      << "ing " << angles.size() << " copies in Service volume"
		      << "\n                            zBend " << zBend 
		      << " zEnd " << zEnd << " rMin " << rMin << " rMax " 
		      << rMax << " Cable width " << width/CLHEP::deg 
		      << " thickness " << thick << " with Angles";
  for (int i=0; i<(int)(angles.size()); i++)
    LogDebug("TIDGeom") << "\tangles[" << i << "] = " << angles[i]/CLHEP::deg;
  LogDebug("TIDGeom") << "                          Wheels " 
		      << zposWheel.size() << " at Z";
  for (int i=0; i<(int)(zposWheel.size()); i++)
    LogDebug("TIDGeom") << "\tzposWheel[" << i <<"] = " << zposWheel[i];
  LogDebug("TIDGeom") << "                          each with " 
		      << zposRing.size() << " Rings at Z";
  for (int i=0; i<(int)(zposRing.size()); i++)
    LogDebug("TIDGeom") << "\tzposRing[" << i <<"] = " << zposRing[i];

  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  matIn       = sArgs["MaterialIn"]; 
  matOut      = sArgs["MaterialOut"]; 

  DDName parentName = parent().name();
  LogDebug("TIDGeom") << "DDTIDAxialCableAlgo debug: Parent " << parentName
		      << "\tChild " << childName << " NameSpace " 
		      << idNameSpace << "\tMaterial " << matIn << " and " 
		      << matOut;
}

void DDTIDAxialCableAlgo::execute(DDCompactView& cpv) {

  DDName mother = parent().name();
  std::vector<DDName> logs;
  double thk = thick/zposRing.size();
  double r   = rMin;
  double thktot = 0;
  double z;

  //Cables between the wheels
  for (int k=0; k<(int)(zposWheel.size()); k++) {

    std::vector<double> pconZ, pconRmin, pconRmax;
    for (int i=0; i<(int)(zposRing.size()); i++) {
      thktot += thk;
      z       = zposWheel[k] + zposRing[i] - 0.5*thk;
      if (i != 0) {
	pconZ.emplace_back(z);
	pconRmin.emplace_back(r);
	pconRmax.emplace_back(rMax);
      }
      r       = rMin;
      pconZ.emplace_back(z);
      pconRmin.emplace_back(r);
      pconRmax.emplace_back(rMax);
      z      += thk;
      pconZ.emplace_back(z);
      pconRmin.emplace_back(r);
      pconRmax.emplace_back(rMax);
      r       = rMax - thktot;
      pconZ.emplace_back(z);
      pconRmin.emplace_back(r);
      pconRmax.emplace_back(rMax);
    }
    if (k >= ((int)(zposWheel.size())-1)) z = zBend;
    else z = zposWheel[k+1] + zposRing[0] - 0.5*thk;
    pconZ.emplace_back(z);
    pconRmin.emplace_back(r);
    pconRmax.emplace_back(rMax);
    
    std::string name = childName + std::to_string(k);
    DDSolid solid = DDSolidFactory::polycone(DDName(name, idNameSpace),
					     -0.5*width, width, pconZ, 
					     pconRmin, pconRmax);

    LogDebug("TIDGeom") << "DDTIDAxialCableAlgo test: " 
			<< DDName(name,idNameSpace) << " Polycone made of "
			<< matIn << " from " << -0.5*width/CLHEP::deg << " to "
			<< 0.5*width/CLHEP::deg << " and with " << pconZ.size()
			<< " sections ";
    for (int i = 0; i <(int)(pconZ.size()); i++) 
      LogDebug("TIDGeom") <<  "\t[" << i  << "]\tZ = " << pconZ[i] 
			  << "\tRmin = "<< pconRmin[i] << "\tRmax = " 
			  << pconRmax[i];

    DDName mat(DDSplit(matIn).first, DDSplit(matIn).second); 
    DDMaterial matter(mat);
    DDLogicalPart genlogic(DDName(name, idNameSpace), matter, solid);
    logs.emplace_back(DDName(name, idNameSpace));
  }

  //Cable in the vertical part
  std::vector<double> pconZ, pconRmin, pconRmax;
  r = thktot*rMax/rTop;
  z = zBend - thktot;
  LogDebug("TIDGeom") << "DDTIDAxialCableAlgo test: Thk " << thk 
		      << " Total " << thktot << " rMax " << rMax 
		      << " rTop " << rTop << " dR " << r << " z " << z;
  pconZ.emplace_back(z);
  pconRmin.emplace_back(rMax);
  pconRmax.emplace_back(rMax);
  z = zBend - r;
  pconZ.emplace_back(z);
  pconRmin.emplace_back(rMax);
  pconRmax.emplace_back(rTop);
  pconZ.emplace_back(zBend);
  pconRmin.emplace_back(rMax);
  pconRmax.emplace_back(rTop);

  std::string name = childName + std::to_string(zposWheel.size());
  DDSolid solid = DDSolidFactory::polycone(DDName(name, idNameSpace),
					   -0.5*width, width, pconZ, 
					   pconRmin, pconRmax);

  LogDebug("TIDGeom") << "DDTIDAxialCableAlgo test: " 
		      << DDName(name, idNameSpace) << " Polycone made of "
		      << matIn << " from " << -0.5*width/CLHEP::deg << " to "
		      << 0.5*width/CLHEP::deg << " and with "  << pconZ.size()
		      << " sections ";
  for (int i = 0; i < (int)(pconZ.size()); i++) 
    LogDebug("TIDGeom") << "\t[" << i << "]\tZ = " << pconZ[i] 
			<< "\tRmin = "<< pconRmin[i] << "\tRmax = " 
			<< pconRmax[i];

  DDMaterial matter(DDName(DDSplit(matIn).first, DDSplit(matIn).second));
  DDLogicalPart genlogic(DDName(name, idNameSpace), matter, solid);
  logs.emplace_back(DDName(name, idNameSpace));

  //Cable in the outer part
  name = childName + std::to_string(zposWheel.size()+1);
  r    = rTop-r;
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 0.5*(zEnd-zBend),
                               r, rTop, -0.5*width, width);
  LogDebug("TIDGeom") << "DDTIDAxialCableAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << matOut << " from " << -0.5*width/CLHEP::deg << " to " 
		      << 0.5*width/CLHEP::deg << " with Rin " << r << " Rout " 
		      << rTop << " ZHalf " << 0.5*(zEnd-zBend);
  matter = DDMaterial(DDName(DDSplit(matOut).first, DDSplit(matOut).second));
  genlogic = DDLogicalPart(DDName(name, idNameSpace), matter, solid);
  logs.emplace_back(DDName(name, idNameSpace));

  //Position the cables
  double theta = 90.*CLHEP::deg;
  for (int i=0; i<(int)(angles.size()); i++) {
    double phix = angles[i];
    double phiy = phix + 90.*CLHEP::deg;
    double phideg = phix/CLHEP::deg;

    DDRotation rotation;
    if (phideg != 0) {
      std::string rotstr = childName + std::to_string(phideg*10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
	LogDebug("TIDGeom") << "DDTIDAxialCableAlgo test: Creating a new "
			    << "rotation: " << rotstr << " " 
			    << theta/CLHEP::deg << ", " << phix/CLHEP::deg 
			    << ", " << theta/CLHEP::deg << ", " 
			    << phiy/CLHEP::deg << ", 0, 0";
	rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, 
			 phiy, 0., 0.);
      }
    }
    
    for (int k=0; k<(int)(logs.size()); k++) {
      DDTranslation tran(0,0,0);
      if (k == ((int)(logs.size())-1))
	tran = DDTranslation(0,0,0.5*(zEnd+zBend));
     cpv.position(logs[k], mother, i+1, tran, rotation);
      LogDebug("TIDGeom") << "DDTIDAxialCableAlgo test " << logs[k] 
			  << " number " << i+1 << " positioned in "
			  << mother << " at " << tran << " with "
			  << rotation;
    }
  }
}
