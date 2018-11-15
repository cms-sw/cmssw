///////////////////////////////////////////////////////////////////////////////
// File: DDTOBAxCableAlgo.cc
// Description: Equipping the axial cylinder of TOB with cables etc
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTOBAxCableAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDTOBAxCableAlgo::DDTOBAxCableAlgo():
  sectorRin(0),sectorRout(0),sectorDeltaPhi_B(0) {
  LogDebug("TOBGeom") <<"DDTOBAxCableAlgo info: Creating an instance";
}

DDTOBAxCableAlgo::~DDTOBAxCableAlgo() {}

void DDTOBAxCableAlgo::initialize(const DDNumericArguments & nArgs,
				  const DDVectorArguments & vArgs,
				  const DDMapArguments &,
				  const DDStringArguments & sArgs,
				  const DDStringVectorArguments & vsArgs) {
  
  idNameSpace  = DDCurrentNamespace::ns();
  LogDebug("TOBGeom") << "DDTOBAxCableAlgo debug: Parent " << parent().name()
		      << " NameSpace " << idNameSpace;
  
  sectorNumber     = vsArgs["SectorNumber"];      
  sectorRin        = nArgs["SectorRin"];
  sectorRout       = nArgs["SectorRout"];
  sectorDz         = nArgs["SectorDz"];
  sectorDeltaPhi_B = nArgs["SectorDeltaPhi_B"];
  sectorStartPhi   = vArgs["SectorStartPhi"];
  sectorMaterial_A = vsArgs["SectorMaterial_A"];
  sectorMaterial_B = vsArgs["SectorMaterial_B"];
  sectorMaterial_C = vsArgs["SectorMaterial_C"];
  
  for (int i=0; i<(int)(sectorNumber.size()); i++)
    LogDebug("TOBGeom") << "DDTOBAxCableAlgo debug: sectorNumber[" << i
			<< "] = " << sectorNumber[i];
  
  LogDebug("TOBGeom") << "DDTOBAxCableAlgo debug: Axial Service Sectors half-length " << sectorDz
		      << "\tRin " << sectorRin << "\tRout = " << sectorRout
		      << "\tPhi of sectors position:";
  for (int i=0; i<(int)(sectorNumber.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tPhi = " << sectorStartPhi[i];
  LogDebug("TOBGeom") << "DDTOBAxCableAlgo debug: List of materials for the sectors/3 parts";
  //
  LogDebug("TOBGeom") << "DDTOBAxCableAlgo debug: Sector/3 A";
  for (int i=0; i<(int)(sectorNumber.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tsectorMaterial_A = " << sectorMaterial_A[i];
  //
  LogDebug("TOBGeom") << "DDTOBAxCableAlgo debug: Sector/3 B";
  for (int i=0; i<(int)(sectorNumber.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tsectorMaterial_B = " << sectorMaterial_B[i];
  //
  LogDebug("TOBGeom") << "DDTOBAxCableAlgo debug: Sector/3 C";
  for (int i=0; i<(int)(sectorNumber.size()); i++)
    LogDebug("TOBGeom") << "\t[" << i << "]\tsectorMaterial_C = " << sectorMaterial_C[i];
  
}

void DDTOBAxCableAlgo::execute(DDCompactView& cpv) {
  
  LogDebug("TOBGeom") << "==>> Constructing DDTOBAxCableAlgo...";
  DDName tubsName(parent().name());
  
  // Loop over sectors (sectorNumber vector)
  for (int i=0; i<(int)(sectorNumber.size()); i++) {
    
    DDSolid solid;
    std::string  name;
    double  dz, rin, rout, startphi, widthphi, deltaphi;
    
    // Axial Services
    // Each sector is divided in 3 parts from phi[i] to phi[i+1]
    
    widthphi = ( (i+1 == (int)(sectorStartPhi.size())) ?
		 (sectorStartPhi[0]+CLHEP::twopi)-sectorStartPhi[i] :
		 (sectorStartPhi[i+1]-sectorStartPhi[i]) );
    
    // First Part: A
    name  = "TOBAxService_" + sectorNumber[i] + "A";
    dz   = sectorDz;
    rin  = sectorRin;
    rout = sectorRout;
    startphi = sectorStartPhi[i];
    deltaphi = 0.5 * (widthphi - sectorDeltaPhi_B);
    
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
				 rout, startphi, deltaphi);
    
    LogDebug("TOBGeom") << "DDTOBAxCableAlgo test: " 
			<< DDName(name, idNameSpace) << " Tubs made of " 
      			<< sectorMaterial_A[i] << " from " 
			<< startphi/CLHEP::deg << " to "
			<< (startphi+deltaphi)/CLHEP::deg << " with Rin " 
			<< rin << " Rout " << rout << " ZHalf " << dz;
    
    DDName sectorMatName(DDSplit(sectorMaterial_A[i]).first,
			 DDSplit(sectorMaterial_A[i]).second);
    DDMaterial sectorMatter(sectorMatName);
    DDLogicalPart sectorLogic(DDName(name, idNameSpace), sectorMatter, solid);
    
    cpv.position(DDName(name,idNameSpace), tubsName, i+1, DDTranslation(), 
	  DDRotation());
    LogDebug("TOBGeom") << "DDTOBAxCableAlgo test: "
			<< DDName(name,idNameSpace) << " number " << i+1 
			<< " positioned in " << tubsName 
			<< " with no translation and no rotation";
    
    // Second Part: B
    name  = "TOBAxService_" + sectorNumber[i] + "B";
    startphi += deltaphi;
    deltaphi = sectorDeltaPhi_B;
    
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
				 rout, startphi, deltaphi);
    
    LogDebug("TOBGeom") << "DDTOBAxCableAlgo test: " 
			<< DDName(name, idNameSpace) << " Tubs made of " 
      			<< sectorMaterial_B[i] << " from " << startphi/CLHEP::deg << " to " << (startphi+deltaphi)/CLHEP::deg
			<< " with Rin " << rin << " Rout " << rout 
			<< " ZHalf " << dz;
    
    sectorMatName = DDName(DDSplit(sectorMaterial_B[i]).first,
			   DDSplit(sectorMaterial_B[i]).second);
    sectorMatter  = DDMaterial(sectorMatName);
    sectorLogic   = DDLogicalPart(DDName(name, idNameSpace), sectorMatter,
				  solid);
    
    cpv.position(DDName(name,idNameSpace), tubsName, i+1, DDTranslation(), 
	  DDRotation());
    LogDebug("TOBGeom") << "DDTOBAxCableAlgo test: "
			<< DDName(name,idNameSpace) << " number " << i+1 
			<< " positioned in " << tubsName 
			<< " with no translation and no rotation";
    
    // Third Part: C
    name  = "TOBAxService_" + sectorNumber[i] + "C";
    startphi += deltaphi;
    deltaphi = 0.5 * (widthphi - sectorDeltaPhi_B);
    
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
				 rout, startphi, deltaphi);
    
    LogDebug("TOBGeom") << "DDTOBAxCableAlgo test: " 
			<< DDName(name, idNameSpace) << " Tubs made of " 
      			<< sectorMaterial_C[i] << " from " 
			<< startphi/CLHEP::deg << " to " 
			<< (startphi+deltaphi)/CLHEP::deg << " with Rin " 
			<< rin << " Rout " << rout << " ZHalf " << dz;
    
    sectorMatName = DDName(DDSplit(sectorMaterial_C[i]).first, 
			   DDSplit(sectorMaterial_C[i]).second);
    sectorMatter  = DDMaterial(sectorMatName);
    sectorLogic   = DDLogicalPart(DDName(name, idNameSpace), sectorMatter, 
				  solid);
    
    cpv.position(DDName(name,idNameSpace), tubsName, i+1, DDTranslation(), 
	  DDRotation());
    LogDebug("TOBGeom") << "DDTOBAxCableAlgo test: "
			<< DDName(name,idNameSpace) << " number " << i+1 
			<< " positioned in " << tubsName 
			<< " with no translation and no rotation";
  }
  
  LogDebug("TOBGeom") << "<<== End of DDTOBAxCableAlgo construction ...";
}
