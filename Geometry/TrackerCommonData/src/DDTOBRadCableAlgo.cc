///////////////////////////////////////////////////////////////////////////////
// File: DDTOBRadCableAlgo.cc
// Description: Equipping the side disks of TOB with cables etc
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/interface/DDTOBRadCableAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTOBRadCableAlgo::DDTOBRadCableAlgo():
  rodRin(0),rodRout(0),cableM(0),connM(0),
  coolR(0),coolRin(0),coolRout1(0),coolRout2(0),
  coolStartPhi1(0),coolDeltaPhi1(0),
  coolStartPhi2(0),coolDeltaPhi2(0),
  names(0) {
  LogDebug("TOBGeom") <<"DDTOBRadCableAlgo info: Creating an instance";
}

DDTOBRadCableAlgo::~DDTOBRadCableAlgo() {}

void DDTOBRadCableAlgo::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments &,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments & vsArgs) {

  idNameSpace  = DDCurrentNamespace::ns();
  DDName parentName = parent().name();
  LogDebug("TOBGeom") << "DDTOBRadCableAlgo debug: Parent " << parentName
		      << " NameSpace " << idNameSpace;

  diskDz       = nArgs["DiskDz"];
  rMax         = nArgs["RMax"];
  cableT       = nArgs["CableT"];     
  rodRin       = vArgs["RodRin"];    
  rodRout      = vArgs["RodRout"];
  cableM       = vsArgs["CableMaterial"];
  LogDebug("TOBGeom") << "DDTOBRadCableAlgo debug: Disk Half width " << diskDz 
		      << "\tRMax " << rMax  << "\tCable Thickness " << cableT 
		      << "\tRadii of disk position and cable materials:";
  for (int i=0; i<(int)(rodRin.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tRin = " << rodRin[i] 
			<< "\tRout = " << rodRout[i] << "  " << cableM[i];

  connW        = nArgs["ConnW"];     
  connT        = nArgs["ConnT"];    
  connM        = vsArgs["ConnMaterial"];
  LogDebug("TOBGeom") << "DDTOBRadCableAlgo debug: Connector Width = " 
		      << connW << "\tThickness = " << connT 
		      << "\tMaterials: ";
  for (int i=0; i<(int)(connM.size()); i++)
    LogDebug("TOBGeom") << "\tconnM[" << i << "] = " << connM[i];

  coolR         = vArgs["CoolR"];    
  coolRin       = nArgs["CoolRin"];
  coolRout1     = nArgs["CoolRout1"];
  coolRout2     = nArgs["CoolRout2"];
  coolStartPhi1 = nArgs["CoolStartPhi1"];
  coolDeltaPhi1 = nArgs["CoolDeltaPhi1"];
  coolStartPhi2 = nArgs["CoolStartPhi2"];
  coolDeltaPhi2 = nArgs["CoolDeltaPhi2"];
  coolM1        = sArgs["CoolMaterial1"];
  coolM2        = sArgs["CoolMaterial2"];
  LogDebug("TOBGeom") << "DDTOBRadCableAlgo debug: Cool Manifold Torus Rin = " << coolRin
		      << " Rout = " << coolRout1
		      << "\t Phi start = " << coolStartPhi1 << " Phi Range = " << coolDeltaPhi1
		      << "\t Material = " << coolM1
		      << "\t Radial positions:";
  for (int i=0; i<(int)(coolR.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i <<"]\tR = " << coolR[i];
  LogDebug("TOBGeom") << "DDTOBRadCableAlgo debug: Cooling Fluid Torus Rin = " << coolRin
		      << " Rout = " << coolRout2
		      << "\t Phi start = " << coolStartPhi2 << " Phi Range = " << coolDeltaPhi2
		      << "\t Material = " << coolM2
		      << "\t Radial positions:";
  for (int i=0; i<(int)(coolR.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i <<"]\tR = " << coolR[i];
  
  names        = vsArgs["RingName"];      
  for (int i=0; i<(int)(names.size()); i++)
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo debug: names[" << i
			<< "] = " << names[i];
}

void DDTOBRadCableAlgo::execute() {
  
  LogDebug("TOBGeom") << "==>> Constructing DDTOBRadCableAlgo...";
  DDName diskName = parent().name();

  // Loop over sub disks
  for (int i=0; i<(int)(names.size()); i++) {

    DDSolid solid;
    std::string  name;
    double  dz, rin, rout;
    
    // Cooling Manifold
    name  = "TOBCoolingManifold" + names[i];
    dz    = coolRout1;
    DDName manifoldName(name, idNameSpace);
    solid = DDSolidFactory::torus(manifoldName,coolRin,coolRout1,coolR[i],coolStartPhi1,coolDeltaPhi1);
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name, idNameSpace) << " Torus made of " 
			<< coolM1 << " from " << coolStartPhi1/deg << " to " << (coolStartPhi1+coolDeltaPhi1)/deg 
			<< " with Rin " << coolRin << " Rout " << coolRout1
			<< " R torus " << coolR[i];
    DDName coolManifoldName(DDSplit(coolM1).first, DDSplit(coolM1).second);
    DDMaterial coolManifoldMatter(coolManifoldName);
    DDLogicalPart coolManifoldLogic(DDName(name, idNameSpace), coolManifoldMatter, solid);
    
    DDTranslation r1(0, 0, (dz-diskDz));
    DDpos(DDName(name,idNameSpace), diskName, i+1, r1, DDRotation());
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name,idNameSpace) << " number " << i+1 
			<< " positioned in " << diskName << " at " << r1
			<< " with no rotation";
    // Cooling Fluid (in Cooling Manifold)
    name  = "TOBCoolingManifoldFluid" + names[i];
    solid = DDSolidFactory::torus(DDName(name, idNameSpace),coolRin,coolRout2,coolR[i],coolStartPhi2,coolDeltaPhi2);
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name, idNameSpace) << " Torus made of " 
			<< coolM2 << " from " << coolStartPhi2/deg << " to " << (coolStartPhi2+coolDeltaPhi2)/deg 
			<< " with Rin " << coolRin << " Rout " << coolRout2
			<< " R torus " << coolR[i];
    DDName coolManifoldFluidName(DDSplit(coolM2).first, DDSplit(coolM2).second);
    DDMaterial coolManifoldFluidMatter(coolManifoldFluidName);
    DDLogicalPart coolManifoldFluidLogic(DDName(name, idNameSpace), coolManifoldFluidMatter, solid);
    DDpos(DDName(name,idNameSpace), manifoldName, i+1, DDTranslation(), DDRotation());
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name,idNameSpace) << " number " << i+1 
			<< " positioned in " << coolManifoldName
			<< " with no translation and no rotation";
    // Connectors
    name  = "TOBConn" + names[i];
    dz    = 0.5*connT;
    rin   = 0.5*(rodRin[i]+rodRout[i])-0.5*connW;
    rout  = 0.5*(rodRin[i]+rodRout[i])+0.5*connW;
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
				 rout, 0, twopi);
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name, idNameSpace) << " Tubs made of " 
			<< connM[i] << " from 0 to " << twopi/deg
			<< " with Rin " << rin << " Rout " << rout 
			<< " ZHalf " << dz;
    DDName connName(DDSplit(connM[i]).first, DDSplit(connM[i]).second);
    DDMaterial connMatter(connName);
    DDLogicalPart connLogic(DDName(name, idNameSpace), connMatter, solid);

    DDTranslation r2(0, 0, (dz-diskDz));
    DDpos(DDName(name,idNameSpace), diskName, i+1, r2, DDRotation());
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name,idNameSpace) << " number " << i+1 
			<< " positioned in " << diskName << " at " << r2 
			<< " with no rotation";

    // Now the radial cable
    name  = "TOBRadCable" + names[i];
    rin   = 0.5*(rodRin[i]+rodRout[i]);
    std::vector<double> pgonZ;
    pgonZ.push_back(-0.5*cableT); 
    pgonZ.push_back(cableT*(rin/rMax-0.5));
    pgonZ.push_back(0.5*cableT);
    std::vector<double> pgonRmin;
    pgonRmin.push_back(rin); 
    pgonRmin.push_back(rin); 
    pgonRmin.push_back(rin); 
    std::vector<double> pgonRmax;
    pgonRmax.push_back(rMax); 
    pgonRmax.push_back(rMax); 
    pgonRmax.push_back(rin); 
    solid = DDSolidFactory::polycone(DDName(name, idNameSpace), 0, twopi,
				     pgonZ, pgonRmin, pgonRmax);
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name, idNameSpace) <<" Polycone made of "
			<< cableM[i] << " from 0 to " << twopi/deg
			<< " and with " << pgonZ.size() << " sections";
    for (int ii = 0; ii < (int)(pgonZ.size()); ii++) 
      LogDebug("TOBGeom") << "\t[" << ii << "]\tZ = " << pgonZ[ii] 
			  << "\tRmin = " << pgonRmin[ii] << "\tRmax = " 
			  << pgonRmax[ii];
    DDName cableName(DDSplit(cableM[i]).first, DDSplit(cableM[i]).second);
    DDMaterial cableMatter(cableName);
    DDLogicalPart cableLogic(DDName(name, idNameSpace), cableMatter, solid);

    DDTranslation r3(0, 0, (diskDz-(i+0.5)*cableT));
    DDpos(DDName(name,idNameSpace), diskName, i+1, r3, DDRotation());
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name,idNameSpace) << " number " <<i+1
			<< " positioned in " << diskName << " at " << r3
			<< " with no rotation";
    
  }

  LogDebug("TOBGeom") << "<<== End of DDTOBRadCableAlgo construction ...";
}
