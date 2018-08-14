///////////////////////////////////////////////////////////////////////////////
// File: DDTIBRadCableAlgo_MTCC.cc
// Description: Equipping the side disks of TIB with cables etc
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/MTCCTrackerCommonData/plugins/DDTIBRadCableAlgo_MTCC.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDTIBRadCableAlgo_MTCC::DDTIBRadCableAlgo_MTCC(): layRin(0),cableMat(0),strucMat(0), layers(0) {
  LogDebug("TIBGeom") << "DDTIBRadCableAlgo_MTCC info: Creating an instance";
}

DDTIBRadCableAlgo_MTCC::~DDTIBRadCableAlgo_MTCC() {}

void DDTIBRadCableAlgo_MTCC::initialize(const DDNumericArguments & nArgs,
					const DDVectorArguments & vArgs,
					const DDMapArguments & ,
					const DDStringArguments & sArgs,
					const DDStringVectorArguments & vsArgs) {

  idNameSpace  = DDCurrentNamespace::ns();
  unsigned int i;
  DDName parentName = parent().name();
  LogDebug("TIBGeom") << "DDTIBRadCableAlgo_MTCC debug: Parent " << parentName 
		      << " NameSpace " << idNameSpace;

  rMin         = nArgs["RMin"];
  rMax         = nArgs["RMax"];
  layRin       = vArgs["RadiusLo"];    
  deltaR       = nArgs["DeltaR"];
  LogDebug("TIBGeom") << "DDTIBRadCableAlgo_MTCC debug: Disk Rmin " << rMin
		      << "\tRMax " << rMax  << "\tSeparation of layers " 
		      << deltaR	<<" with " << layRin.size() <<" layers at R =";
  for (i = 0; i < layRin.size(); i++)
    LogDebug("TIBGeom") << " " << i << " : " << layRin[i];
  
  cylinderT    = nArgs["CylinderThick"];
  supportT     = nArgs["SupportThick"];
  supportDR    = nArgs["SupportDR"];
  supportMat   = sArgs["SupportMaterial"];
  LogDebug("TIBGeom") << "DDTIBRadCableAlgo_MTCC debug: SupportCylinder "
		      << "Thickness " << cylinderT << "\tSupportDisk "
		      << "Thickness " << supportT << "\tExtra width along R " 
		      << supportDR << "\tMaterial: " << supportMat;

  cableT       = nArgs["CableThick"];     
  cableMat     = vsArgs["CableMaterial"];
  LogDebug("TIBGeom") << "DDTIBRadCableAlgo_MTCC debug: Cable Thickness "
		      << cableT  << " with materials: ";
  for (i = 0; i < cableMat.size(); i++)
    LogDebug("TIBGeom") << " " << i << " : " << cableMat[i];

  strucMat     = vsArgs["StructureMaterial"];
  LogDebug("TIBGeom") << "DDTIBRadCableAlgo_MTCC debug: " << strucMat.size()
		      << " materials for open structure:";
  for (i=0; i<strucMat.size(); i++)
    LogDebug("TIBGeom") << " " << i << " : " << strucMat[i];
  
  layers       = vArgs["Layers"];
  LogDebug("TIBGeom") << "DDTIBRadCableAlgo_MTCC debug: " << layers.size()
		      << " layers:";
  for (i=0; i<layers.size(); i++)
    LogDebug("TIBGeom") << " " << i << " : " << layers[i];
  
}

void DDTIBRadCableAlgo_MTCC::execute(DDCompactView& cpv) {
  
  LogDebug("TIBGeom") << "==>> Constructing DDTIBRadCableAlgo_MTCC...";
  DDName diskName = parent().name();

  DDSolid solid;
  std::string  name;
  double  rin, rout;

  // Loop over sub disks
  DDName suppName(DDSplit(supportMat).first, DDSplit(supportMat).second);
  DDMaterial suppMatter(suppName);
  double diskDz = 0.5 * (supportT + cableT*layRin.size());
  double dz     = 0.5*supportT;

  for (unsigned int i=0; i<layRin.size(); i++) {

    // fill only layers in layers list
    bool empty=true;
    for(double layer : layers) {
      if(i+1==(unsigned int)layer) {
	empty=false;
      }
    }
    
    if(!empty) {
      //Support disks
      name  = "TIBSupportSideDisk" + std::to_string(i);
      rin   = layRin[i]+0.5*(deltaR-cylinderT)-supportDR;
      rout  = layRin[i]+0.5*(deltaR+cylinderT)+supportDR;
      solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
				   rout, 0, CLHEP::twopi);
      LogDebug("TIBGeom") << "DDTIBRadCableAlgo_MTCC test: " 
			  << DDName(name, idNameSpace) << " Tubs made of "
			  << supportMat << " from 0 to " 
			  << CLHEP::twopi/CLHEP::deg << " with Rin " << rin 
			  << " Rout " << rout << " ZHalf " << dz;
      DDLogicalPart suppLogic(DDName(name, idNameSpace), suppMatter, solid);
      
      DDTranslation r1(0, 0, (dz-diskDz));
      cpv.position(DDName(name,idNameSpace), diskName, i+1, r1, DDRotation());
      LogDebug("TIBGeom") << "DDTIBRadCableAlgo_MTCC test: " 
			  << DDName(name,idNameSpace) << " number " << i+1 
			  << " positioned in " << diskName << " at " << r1 
			  << " with no rotation";
      
      //Open Structure
      name  = "TIBOpenZone" + std::to_string(i);
      rout  = rin;
      if (i == 0) rin = rMin;
      else        rin = layRin[i-1]+0.5*(deltaR+cylinderT)+supportDR;
      solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
				   rout, 0, CLHEP::twopi);
      LogDebug("TIBGeom") << "DDTIBRadCableAlgo_MTCC test: "
			  << DDName(name, idNameSpace) << " Tubs made of " 
			  << strucMat[i] << " from 0 to "
			  << CLHEP::twopi/CLHEP::deg << " with Rin " << rin 
			  << " Rout " << rout << " ZHalf " << dz;
      DDName strucName(DDSplit(strucMat[i]).first, DDSplit(strucMat[i]).second);
      DDMaterial strucMatter(strucName);
      DDLogicalPart strucLogic(DDName(name, idNameSpace), strucMatter, solid);
      
      DDTranslation r2(0, 0, (dz-diskDz));
      cpv.position(DDName(name,idNameSpace), diskName, i+1, r2, DDRotation());
      LogDebug("TIBGeom") << "DDTIBRadCableAlgo_MTCC test: " 
			  << DDName(name,idNameSpace) << " number " << i+1 
			  << " positioned in " << diskName 
			  << " at " << r2 << " with no rotation";
      
      //Now the radial cable
      name  = "TIBRadCable" + std::to_string(i);
      double rv = layRin[i]+0.5*deltaR;
      std::vector<double> pgonZ;
      pgonZ.emplace_back(-0.5*cableT); 
      pgonZ.emplace_back(cableT*(rv/rMax-0.5));
      pgonZ.emplace_back(0.5*cableT);
      std::vector<double> pgonRmin;
      pgonRmin.emplace_back(rv); 
      pgonRmin.emplace_back(rv); 
      pgonRmin.emplace_back(rv); 
      std::vector<double> pgonRmax;
      pgonRmax.emplace_back(rMax); 
      pgonRmax.emplace_back(rMax); 
      pgonRmax.emplace_back(rv); 
      solid = DDSolidFactory::polycone(DDName(name, idNameSpace), 0,
				       CLHEP::twopi,pgonZ, pgonRmin, pgonRmax);
      LogDebug("TIBGeom") << "DDTIBRadCableAlgo_MTCC test: "
			  << DDName(name, idNameSpace) 
			  << " Polycone made of " << cableMat[i]
			  << " from 0 to " << CLHEP::twopi/CLHEP::deg 
			  << " and with " << pgonZ.size() << " sections";
      for (unsigned int ii = 0; ii <pgonZ.size(); ii++) 
	LogDebug("TIBGeom") << "\t" << "\tZ = " << pgonZ[ii] << "\tRmin = " 
			    << pgonRmin[ii] << "\tRmax = " << pgonRmax[ii];
      DDName cableName(DDSplit(cableMat[i]).first,DDSplit(cableMat[i]).second);
      LogDebug("TIBGeom") << " material cableName " << i << " " << cableName;
      DDMaterial cableMatter(cableName);
      DDLogicalPart cableLogic(DDName(name, idNameSpace), cableMatter, solid);
      
      DDTranslation r3(0, 0, (diskDz-(i+0.5)*cableT));
      cpv.position(DDName(name,idNameSpace), diskName, i+1, r3, DDRotation());
      LogDebug("TIBGeom") << "DDTIBRadCableAlgo_MTCC test: " 
			  << DDName(name,idNameSpace) << " number " << i+1
			  << " positioned in " << diskName << " at "
			  << r3 << " with no rotation";
      
    }
    //
    
    // fill only layers in layers list
    empty=true;
    for(double layer : layers) {
      if(i+1==(unsigned int)layer) {
	empty=false;
      }
    }
    
    if(!empty) {
      //Now the last open zone
      unsigned int i = layRin.size();
      rin  = layRin[i-1]+0.5*(deltaR+cylinderT)+supportDR;
      rout = rMax;
      name  = "TIBOpenZone" + std::to_string(i);
      solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
				   rout, 0, CLHEP::twopi);
      LogDebug("TIBGeom") << "DDTIBRadCableAlgo_MTCC test: "
			  << DDName(name, idNameSpace) << " Tubs made of "
			  << strucMat[i] << " from 0 to " 
			  << CLHEP::twopi/CLHEP::deg << " with Rin " << rin 
			  << " Rout " << rout << " ZHalf " << dz;
      DDName strucName(DDSplit(strucMat[i]).first,DDSplit(strucMat[i]).second);
      DDMaterial strucMatter(strucName);
      DDLogicalPart strucLogic(DDName(name, idNameSpace), strucMatter, solid);
      
      DDTranslation r2(0, 0, (dz-diskDz));
      cpv.position(DDName(name,idNameSpace), diskName, i+1, r2, DDRotation());
      LogDebug("TIBGeom") << "DDTIBRadCableAlgo_MTCC test: "
			  << DDName(name,idNameSpace) << " number " << i+1 
			  << " positioned in " << diskName 
			  << " at " << r2 << " with no rotation";
    } 
  }
  LogDebug("TIBGeom") << "<<== End of DDTIBRadCableAlgo_MTCC construction ...";
 
}
