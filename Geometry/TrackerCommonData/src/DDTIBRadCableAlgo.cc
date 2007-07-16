///////////////////////////////////////////////////////////////////////////////
// File: DDTIBRadCableAlgo.cc
// Description: Equipping the side disks of TIB with cables etc
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/interface/DDTIBRadCableAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTIBRadCableAlgo::DDTIBRadCableAlgo(): layRin(0),strucMat(0) {
  LogDebug("TIBGeom") <<"DDTIBRadCableAlgo info: Creating an instance";
}

DDTIBRadCableAlgo::~DDTIBRadCableAlgo() {}

void DDTIBRadCableAlgo::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments & vsArgs) {

  idNameSpace  = DDCurrentNamespace::ns();
  DDName parentName = parent().name();
  LogDebug("TIBGeom") << "DDTIBRadCableAlgo debug: Parent " << parentName 
		      << " NameSpace " << idNameSpace;

  rMin         = nArgs["RMin"];
  rMax         = nArgs["RMax"];
  layRin       = vArgs["RadiusLo"];    
  deltaR       = nArgs["DeltaR"];
  LogDebug("TIBGeom") << "DDTIBRadCableAlgo debug: Disk Rmin " << rMin 
		      << "\tRMax " << rMax  << "\tSeparation of layers "
		      << deltaR	<< " with " << layRin.size() 
		      << " layers at R =";
  for (int i = 0; i < (int)(layRin.size()); i++)
    LogDebug("TIBGeom") << "\tLayRin[" << i << "] = " << layRin[i];

  cylinderT    = nArgs["CylinderThick"];
  supportT     = nArgs["SupportThick"];
  supportDR    = nArgs["SupportDR"];
  supportMat   = sArgs["SupportMaterial"];
  LogDebug("TIBGeom") << "DDTIBRadCableAlgo debug: SupportCylinder "
		      << " Thickness " << cylinderT << "\tSupportDisk "
		      << "Thickness " << supportT << "\tExtra width along "
		      << "R " << supportDR << "\tMaterial: " << supportMat;

  strucMat     = vsArgs["StructureMaterial"];
  LogDebug("TIBGeom") << "DDTIBRadCableAlgo debug: " << strucMat.size()
		      << " materials for open structure:";
  for (int i = 0; i < (int)(strucMat.size()); i++)
    LogDebug("TIBGeom") << "\tstrucMat[" << i << "] = " << strucMat[i];
}

void DDTIBRadCableAlgo::execute() {
  
  LogDebug("TIBGeom") << "==>> Constructing DDTIBRadCableAlgo...";
  DDName diskName = parent().name();

  DDSolid solid;
  std::string  name;
  double  rin, rout;
  
  // Loop over sub disks
  DDName suppName(DDSplit(supportMat).first, DDSplit(supportMat).second);
  DDMaterial suppMatter(suppName);
  double diskDz = 0.5*supportT;
  double dz     = 0.5*supportT;

  for (int i=0; i<(int)(layRin.size()); i++) {

    //Support disks
    name  = "TIBSupportSideDisk" + dbl_to_string(i);
    rin   = layRin[i]+0.5*(deltaR-cylinderT)-supportDR;
    rout  = layRin[i]+0.5*(deltaR+cylinderT)+supportDR;
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
				 rout, 0, twopi);
    LogDebug("TIBGeom") << "DDTIBRadCableAlgo test: " 
			<< DDName(name, idNameSpace) << " Tubs made of "
			<< supportMat << " from 0 to " << twopi/deg 
			<< " with Rin " << rin << " Rout " << rout 
			<< " ZHalf " << dz;
    DDLogicalPart suppLogic(DDName(name, idNameSpace), suppMatter, solid);

    DDTranslation r1(0, 0, (dz-diskDz));
    DDpos(DDName(name,idNameSpace), diskName, i+1, r1, DDRotation());
    LogDebug("TIBGeom") << "DDTIBRadCableAlgo test: " 
			<< DDName(name,idNameSpace) << " number " << i+1
			<< " positioned in " << diskName << " at " << r1 
			<< " with no rotation";
    
    //Open Structure
    name  = "TIBOpenZone" + dbl_to_string(i);
    rout  = rin;
    if (i == 0) rin = rMin;
    else        rin = layRin[i-1]+0.5*(deltaR+cylinderT)+supportDR;
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
				 rout, 0, twopi);
    LogDebug("TIBGeom") << "DDTIBRadCableAlgo test: "
			<< DDName(name, idNameSpace) << " Tubs made of "
			<< strucMat[i] << " from 0 to " << twopi/deg
			<< " with Rin " << rin << " Rout " << rout 
			<< " ZHalf " << dz;
    DDName strucName(DDSplit(strucMat[i]).first, DDSplit(strucMat[i]).second);
    DDMaterial strucMatter(strucName);
    DDLogicalPart strucLogic(DDName(name, idNameSpace), strucMatter, solid);

    DDTranslation r2(0, 0, (dz-diskDz));
    DDpos(DDName(name,idNameSpace), diskName, i+1, r2, DDRotation());
    LogDebug("TIBGeom") << "DDTIBRadCableAlgo test: "
			<< DDName(name,idNameSpace) << " number " << i+1
			<< " positioned in " << diskName << " at " << r2
			<< " with no rotation";
    
  }

  //Now the last open zone
  unsigned int i = layRin.size();
  rin  = layRin[i-1]+0.5*(deltaR+cylinderT)+supportDR;
  rout = rMax;
  name  = "TIBOpenZone" + dbl_to_string(i);
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
			       rout, 0, twopi);
  LogDebug("TIBGeom") << "DDTIBRadCableAlgo test: " 
		      << DDName(name, idNameSpace) << " Tubs made of " 
		      << strucMat[i] << " from 0 to " << twopi/deg
		      << " with Rin " << rin << " Rout " << rout 
		      << " ZHalf " << dz;
  DDName strucName(DDSplit(strucMat[i]).first, DDSplit(strucMat[i]).second);
  DDMaterial strucMatter(strucName);
  DDLogicalPart strucLogic(DDName(name, idNameSpace), strucMatter, solid);

  DDTranslation r2(0, 0, (dz-diskDz));
  DDpos(DDName(name,idNameSpace), diskName, i+1, r2, DDRotation());
  LogDebug("TIBGeom") << "DDTIBRadCableAlgo test: " 
		      << DDName(name,idNameSpace) << " number " << i+1
		      << " positioned in " << diskName << " at " << r2
		      << " with no rotation";

  LogDebug("TIBGeom") << "<<== End of DDTIBRadCableAlgo construction ...";
}
