///////////////////////////////////////////////////////////////////////////////
// File: DDTOBRadCableAlgo.cc
// Description: Equipping the side disks of TOB with cables etc
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTOBRadCableAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDTOBRadCableAlgo::DDTOBRadCableAlgo():
  rodRin(0),rodRout(0),cableM(0),connM(0),
  coolR1(0),coolR2(0),coolRin(0),coolRout1(0),coolRout2(0),
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
  DDCurrentNamespace ns;
  idNameSpace  = *ns;
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

  coolR1        = vArgs["CoolR1"];
  coolR2        = vArgs["CoolR2"];
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
  for (int i=0; i<(int)(coolR1.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i <<"]\tR = " << coolR1[i];
  for (int i=0; i<(int)(coolR2.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i <<"]\tR = " << coolR2[i];
  LogDebug("TOBGeom") << "DDTOBRadCableAlgo debug: Cooling Fluid Torus Rin = " << coolRin
		      << " Rout = " << coolRout2
		      << "\t Phi start = " << coolStartPhi2 << " Phi Range = " << coolDeltaPhi2
		      << "\t Material = " << coolM2
		      << "\t Radial positions:";
  for (int i=0; i<(int)(coolR1.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i <<"]\tR = " << coolR1[i];
  for (int i=0; i<(int)(coolR2.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i <<"]\tR = " << coolR2[i];
  
  names        = vsArgs["RingName"];      
  for (int i=0; i<(int)(names.size()); i++)
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo debug: names[" << i
			<< "] = " << names[i];
}

void DDTOBRadCableAlgo::execute(DDCompactView& cpv) {
  
  LogDebug("TOBGeom") << "==>> Constructing DDTOBRadCableAlgo...";
  DDName diskName = parent().name();

  // Loop over sub disks
  for (int i=0; i<(int)(names.size()); i++) {

    DDSolid solid;
    std::string  name;
    double  dz, rin, rout;
    
    // Cooling Manifolds
    name  = "TOBCoolingManifold" + names[i] + "a";
    dz    = coolRout1;
    DDName manifoldName_a(name, idNameSpace);
    solid = DDSolidFactory::torus(manifoldName_a,coolRin,coolRout1,coolR1[i],coolStartPhi1,coolDeltaPhi1);
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name, idNameSpace) << " Torus made of " 
			<< coolM1 << " from " << coolStartPhi1/CLHEP::deg 
			<< " to " << (coolStartPhi1+coolDeltaPhi1)/CLHEP::deg 
			<< " with Rin " << coolRin << " Rout " << coolRout1
			<< " R torus " << coolR1[i];
    DDName coolManifoldName_a(DDSplit(coolM1).first, DDSplit(coolM1).second);
    DDMaterial coolManifoldMatter_a(coolManifoldName_a);
    DDLogicalPart coolManifoldLogic_a(DDName(name, idNameSpace), 
				      coolManifoldMatter_a, solid);
    
    DDTranslation r1(0, 0, (dz-diskDz));
    cpv.position(DDName(name,idNameSpace), diskName, i+1, r1, DDRotation());
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name,idNameSpace) << " number " << i+1 
			<< " positioned in " << diskName << " at " << r1
			<< " with no rotation";
    // Cooling Fluid (in Cooling Manifold)
    name  = "TOBCoolingManifoldFluid" + names[i] + "a";
    solid = DDSolidFactory::torus(DDName(name, idNameSpace),coolRin,coolRout2,
				  coolR1[i],coolStartPhi2,coolDeltaPhi2);
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name, idNameSpace) << " Torus made of " 
			<< coolM2 << " from " << coolStartPhi2/CLHEP::deg 
			<< " to " << (coolStartPhi2+coolDeltaPhi2)/CLHEP::deg 
			<< " with Rin " << coolRin << " Rout " << coolRout2
			<< " R torus " << coolR1[i];
    DDName coolManifoldFluidName_a(DDSplit(coolM2).first, 
				   DDSplit(coolM2).second);
    DDMaterial coolManifoldFluidMatter_a(coolManifoldFluidName_a);
    DDLogicalPart coolManifoldFluidLogic_a(DDName(name, idNameSpace),
					   coolManifoldFluidMatter_a, solid);
    cpv.position(DDName(name,idNameSpace), manifoldName_a, i+1, DDTranslation(),
	  DDRotation());
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name,idNameSpace) << " number " << i+1 
			<< " positioned in " << coolManifoldName_a
			<< " with no translation and no rotation";
    //
    name  = "TOBCoolingManifold" + names[i] + "r";
    dz    = coolRout1;
    DDName manifoldName_r(name, idNameSpace);
    solid = DDSolidFactory::torus(manifoldName_r,coolRin,coolRout1,coolR2[i],
				  coolStartPhi1,coolDeltaPhi1);
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name, idNameSpace) << " Torus made of " 
			<< coolM1 << " from " << coolStartPhi1/CLHEP::deg 
			<< " to " << (coolStartPhi1+coolDeltaPhi1)/CLHEP::deg 
			<< " with Rin " << coolRin << " Rout " << coolRout1
			<< " R torus " << coolR2[i];
    DDName coolManifoldName_r(DDSplit(coolM1).first, DDSplit(coolM1).second);
    DDMaterial coolManifoldMatter_r(coolManifoldName_r);
    DDLogicalPart coolManifoldLogic_r(DDName(name, idNameSpace), 
				      coolManifoldMatter_r, solid);
    
    r1 = DDTranslation(0, 0, (dz-diskDz));
    cpv.position(DDName(name,idNameSpace), diskName, i+1, r1, DDRotation());
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name,idNameSpace) << " number " << i+1 
			<< " positioned in " << diskName << " at " << r1
			<< " with no rotation";
    // Cooling Fluid (in Cooling Manifold)
    name  = "TOBCoolingManifoldFluid" + names[i] + "r";
    solid = DDSolidFactory::torus(DDName(name, idNameSpace),coolRin,coolRout2,
				  coolR2[i],coolStartPhi2,coolDeltaPhi2);
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name, idNameSpace) << " Torus made of " 
			<< coolM2 << " from " << coolStartPhi2/CLHEP::deg 
			<< " to " << (coolStartPhi2+coolDeltaPhi2)/CLHEP::deg 
			<< " with Rin " << coolRin << " Rout " << coolRout2
			<< " R torus " << coolR2[i];
    DDName coolManifoldFluidName_r(DDSplit(coolM2).first, 
				   DDSplit(coolM2).second);
    DDMaterial coolManifoldFluidMatter_r(coolManifoldFluidName_r);
    DDLogicalPart coolManifoldFluidLogic_r(DDName(name, idNameSpace), 
					   coolManifoldFluidMatter_r, solid);
    cpv.position(DDName(name,idNameSpace), manifoldName_r, i+1, DDTranslation(), 
	  DDRotation());
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name,idNameSpace) << " number " << i+1 
			<< " positioned in " << coolManifoldName_r
			<< " with no translation and no rotation";
    
    // Connectors
    name  = "TOBConn" + names[i];
    dz    = 0.5*connT;
    rin   = 0.5*(rodRin[i]+rodRout[i])-0.5*connW;
    rout  = 0.5*(rodRin[i]+rodRout[i])+0.5*connW;
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
				 rout, 0, CLHEP::twopi);
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name, idNameSpace) << " Tubs made of " 
			<< connM[i] << " from 0 to " << CLHEP::twopi/CLHEP::deg
			<< " with Rin " << rin << " Rout " << rout 
			<< " ZHalf " << dz;
    DDName connName(DDSplit(connM[i]).first, DDSplit(connM[i]).second);
    DDMaterial connMatter(connName);
    DDLogicalPart connLogic(DDName(name, idNameSpace), connMatter, solid);

    DDTranslation r2(0, 0, (dz-diskDz));
    cpv.position(DDName(name,idNameSpace), diskName, i+1, r2, DDRotation());
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name,idNameSpace) << " number " << i+1 
			<< " positioned in " << diskName << " at " << r2 
			<< " with no rotation";

    // Now the radial cable
    name  = "TOBRadServices" + names[i];
    rin   = 0.5*(rodRin[i]+rodRout[i]);
    rout = ( i+1 == (int)(names.size()) ? rMax : 0.5*(rodRin[i+1]+rodRout[i+1]));
    std::vector<double> pgonZ;
    pgonZ.emplace_back(-0.5*cableT); 
    pgonZ.emplace_back(cableT*(rin/rMax-0.5));
    pgonZ.emplace_back(0.5*cableT);
    std::vector<double> pgonRmin;
    pgonRmin.emplace_back(rin); 
    pgonRmin.emplace_back(rin); 
    pgonRmin.emplace_back(rin); 
    std::vector<double> pgonRmax;
    pgonRmax.emplace_back(rout); 
    pgonRmax.emplace_back(rout); 
    pgonRmax.emplace_back(rout); 
    solid = DDSolidFactory::polycone(DDName(name,idNameSpace), 0, CLHEP::twopi,
				     pgonZ, pgonRmin, pgonRmax);
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name, idNameSpace) <<" Polycone made of "
			<< cableM[i] << " from 0 to " 
			<< CLHEP::twopi/CLHEP::deg << " and with " 
			<< pgonZ.size() << " sections";
    for (int ii = 0; ii < (int)(pgonZ.size()); ii++) 
      LogDebug("TOBGeom") << "\t[" << ii << "]\tZ = " << pgonZ[ii] 
			  << "\tRmin = " << pgonRmin[ii] << "\tRmax = " 
			  << pgonRmax[ii];
    DDName cableName(DDSplit(cableM[i]).first, DDSplit(cableM[i]).second);
    DDMaterial cableMatter(cableName);
    DDLogicalPart cableLogic(DDName(name, idNameSpace), cableMatter, solid);

    DDTranslation r3(0, 0, (diskDz-(i+0.5)*cableT));
    cpv.position(DDName(name,idNameSpace), diskName, i+1, r3, DDRotation());
    LogDebug("TOBGeom") << "DDTOBRadCableAlgo test: " 
			<< DDName(name,idNameSpace) << " number " <<i+1
			<< " positioned in " << diskName << " at " << r3
			<< " with no rotation";
    
  }

  LogDebug("TOBGeom") << "<<== End of DDTOBRadCableAlgo construction ...";
}
