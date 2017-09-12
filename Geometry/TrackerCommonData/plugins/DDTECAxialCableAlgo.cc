///////////////////////////////////////////////////////////////////////////////
// File: DDTECAxialCableAlgo.cc
// Description: Position n copies at prescribed phi values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTECAxialCableAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDTECAxialCableAlgo::DDTECAxialCableAlgo() {
  LogDebug("TECGeom") << "DDTECAxialCableAlgo info: Creating an instance";
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
  
  if (fabs(rangeAngle-360.0*CLHEP::deg)<0.001*CLHEP::deg) { 
    delta    =   rangeAngle/double(n);
  } else {
    if (n > 1) {
      delta  =   rangeAngle/double(n-1);
    } else {
      delta  = 0.;
    }
  }  

  LogDebug("TECGeom") << "DDTECAxialCableAlgo debug: Parameters for creating " 
		      << startAngle.size() << " axial cables and positioning "
		      << n << " copies in Service volume\n"
		      << "                            zStart " << zStart 
		      << " zEnd " << zEnd << " rMin " << rMin << " rMax "
		      << rMax << " Cable width " << width/CLHEP::deg 
		      << " thickness " << thickR << ", " << thickZ << " dZ " 
		      << dZ << "\n                            Range, Delta " 
		      << rangeAngle/CLHEP::deg << ", " << delta/CLHEP::deg;
  for (int i=0; i<(int)(startAngle.size()); i++)
    LogDebug("TECGeom") << "                          Cable " << i 
			<< " from Z " << zPos[i] << " startAngle " 
			<< startAngle[i]/CLHEP::deg;
  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  matName     = sArgs["Material"]; 

  DDName parentName = parent().name();

  LogDebug("TECGeom") << "DDTECAxialCableAlgo debug: Parent " << parentName
		      << "\tChild " << childName << " NameSpace " 
		      << idNameSpace << "\tMaterial " << matName;
}

void DDTECAxialCableAlgo::execute(DDCompactView& cpv) {

  DDName mother = parent().name();
  double theta  = 90.*CLHEP::deg;

  for (int k=0; k<(int)(startAngle.size()); k++) {

    int i;
    double zv = zPos[k]-dZ-0.5*(zStart+zEnd);
    std::vector<double> pconZ, pconRmin, pconRmax;
    pconZ.emplace_back(zv);
    pconRmin.emplace_back(rMin);
    pconRmax.emplace_back(rMax);
    pconZ.emplace_back(zv+thickZ);
    pconRmin.emplace_back(rMin);
    pconRmax.emplace_back(rMax);
    pconZ.emplace_back(zv+thickZ);
    pconRmin.emplace_back(rMax-thickR);
    pconRmax.emplace_back(rMax);
    zv = zPos[k]+dZ-0.5*(zStart+zEnd);
    pconZ.emplace_back(zv-thickZ);
    pconRmin.emplace_back(rMax-thickR);
    pconRmax.emplace_back(rMax);
    pconZ.emplace_back(zv-thickZ);
    pconRmin.emplace_back(rMin);
    pconRmax.emplace_back(rMax);
    pconZ.emplace_back(zv);
    pconRmin.emplace_back(rMin);
    pconRmax.emplace_back(rMax);
    pconZ.emplace_back(zv);
    pconRmin.emplace_back(rMax-thickR);
    pconRmax.emplace_back(rMax);
    pconZ.emplace_back(0.5*(zEnd-zStart));
    pconRmin.emplace_back(rMax-thickR);
    pconRmax.emplace_back(rMax);

    std::string name = childName + std::to_string(k);
    DDSolid solid = DDSolidFactory::polycone(DDName(name, idNameSpace),
					     -0.5*width, width, pconZ, 
					     pconRmin, pconRmax);

    LogDebug("TECGeom") << "DDTECAxialCableAlgo test: " 
			<< DDName(name, idNameSpace) <<" Polycone made of "
			<< matName << " from " <<-0.5*width/CLHEP::deg <<" to "
			<< 0.5*width/CLHEP::deg << " and with " << pconZ.size()
			<< " sections ";
    for (int ii = 0; ii <(int)(pconZ.size()); ii++) 
      LogDebug("TECGeom") << "\t" << "\tZ[" << ii << "] = " << pconZ[ii] 
			  << "\tRmin[" << ii << "] = "<< pconRmin[ii] 
			  << "\tRmax[" << ii << "] = " << pconRmax[ii];
    DDName mat(DDSplit(matName).first, DDSplit(matName).second); 
    DDMaterial matter(mat);
    DDLogicalPart genlogic(DDName(name, idNameSpace), matter, solid);
    
    double phi = startAngle[k];
    for (i=0; i<n; i++) {
      double phix = phi;
      double phiy = phix + 90.*CLHEP::deg;
      double phideg = phix/CLHEP::deg;

      DDRotation rotation;
      if (phideg != 0) {
	std::string rotstr = childName + std::to_string(phideg*10.);
	rotation = DDRotation(DDName(rotstr, idNameSpace));
	if (!rotation) {
	  LogDebug("TECGeom") << "DDTECAxialCableAlgo test: Creating a new"
			      << " rotation: " << rotstr << "\t90., " 
			      << phix/CLHEP::deg << ", 90.," 
			      << phiy/CLHEP::deg << ", 0, 0";
	  rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, 
			   phiy, 0., 0.);
	}
      }
	
      DDTranslation tran(0,0,0);
     cpv.position(DDName(name, idNameSpace), mother, i+1, tran, rotation);
      LogDebug("TECGeom") << "DDTECAxialCableAlgo test " 
			  << DDName(name, idNameSpace) << " number " << i+1
			  << " positioned in " << mother << " at " << tran
			  << " with "  << rotation;

      phi  += delta;
    }
  }
}
