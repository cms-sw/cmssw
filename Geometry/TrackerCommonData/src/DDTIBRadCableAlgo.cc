#define DEBUG 0
#define COUT if (DEBUG) cout
///////////////////////////////////////////////////////////////////////////////
// File: DDTIBRadCableAlgo.cc
// Description: Equipping the side disks of TIB with cables etc
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "Geometry/TrackerSimData/interface/DDTIBRadCableAlgo.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTIBRadCableAlgo::DDTIBRadCableAlgo(): layRin(0),cableMat(0),strucMat(0) {
  COUT << "DDTIBRadCableAlgo info: Creating an instance" << endl;
}

DDTIBRadCableAlgo::~DDTIBRadCableAlgo() {}

void DDTIBRadCableAlgo::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments & vsArgs) {

  idNameSpace  = DDCurrentNamespace::ns();
  unsigned int i;
  DDName parentName = parent().name();
  COUT << "DDTIBRadCableAlgo debug: Parent " << parentName 
		<< " NameSpace " << idNameSpace << endl;

  rMin         = nArgs["RMin"];
  rMax         = nArgs["RMax"];
  layRin       = vArgs["RadiusLo"];    
  deltaR       = nArgs["DeltaR"];
  COUT << "DDTIBRadCableAlgo debug: Disk Rmin " << rMin
		<< "\tRMax " << rMax  << "\tSeparation of layers " << deltaR
		<< " with " << layRin.size() << " layers at R =";
  for (i = 0; i < layRin.size(); i++)
    COUT << " " << layRin[i];
  COUT << endl;

  cylinderT    = nArgs["CylinderThick"];
  supportT     = nArgs["SupportThick"];
  supportDR    = nArgs["SupportDR"];
  supportMat   = sArgs["SupportMaterial"];
  COUT << "DDTIBRadCableAlgo debug: SupportCylinder Thickness " 
		<< cylinderT << "\tSupportDisk Thickness " << supportT 
		<< "\tExtra width along R " << supportDR
		<< "\tMaterial: " << supportMat << endl;

  cableT       = nArgs["CableThick"];     
  cableMat     = vsArgs["CableMaterial"];
  COUT << "DDTIBRadCableAlgo debug: Cable Thickness " << cableT  
		<< " with materials: ";
  for (i = 0; i < cableMat.size(); i++)
    COUT << " " << cableMat[i];
  COUT << endl;

  strucMat     = vsArgs["StructureMaterial"];
  COUT << "DDTIBRadCableAlgo debug: " << strucMat.size()
		<< " materials for open structure:";
  for (i=0; i<strucMat.size(); i++)
    COUT << " " << strucMat[i];
  COUT << endl;
}

void DDTIBRadCableAlgo::execute() {
  
  COUT << "==>> Constructing DDTIBRadCableAlgo..." << endl;
  DDName diskName = parent().name();

  DDSolid solid;
  string  name;
  double  rin, rout;

  // Loop over sub disks
  DDName suppName(DDSplit(supportMat).first, DDSplit(supportMat).second);
  DDMaterial suppMatter(suppName);
  double diskDz = 0.5 * (supportT + cableT*layRin.size());
  double dz     = 0.5*supportT;

  for (unsigned int i=0; i<layRin.size(); i++) {

    //Support disks
    name  = "TIBSupportSideDisk" + dbl_to_string(i);
    rin   = layRin[i]+0.5*(deltaR-cylinderT)-supportDR;
    rout  = layRin[i]+0.5*(deltaR+cylinderT)+supportDR;
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
				 rout, 0, twopi);
    COUT << "DDTIBRadCableAlgo test: " << DDName(name, idNameSpace) 
		 << " Tubs made of " << supportMat << " from 0 to " 
		 << twopi/deg << " with Rin " << rin << " Rout " << rout 
		 << " ZHalf " << dz << endl;
    DDLogicalPart suppLogic(DDName(name, idNameSpace), suppMatter, solid);

    DDTranslation r1(0, 0, (dz-diskDz));
    DDpos(DDName(name,idNameSpace), diskName, i+1, r1, DDRotation());
    COUT << "DDTIBRadCableAlgo test: " << DDName(name,idNameSpace) 
		 << " number " << i+1 << " positioned in " << diskName 
		 << " at " << r1 << " with no rotation" << endl;
    
    //Open Structure
    name  = "TIBOpenZone" + dbl_to_string(i);
    rout  = rin;
    if (i == 0) rin = rMin;
    else        rin = layRin[i-1]+0.5*(deltaR+cylinderT)+supportDR;
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
				 rout, 0, twopi);
    COUT << "DDTIBRadCableAlgo test: " << DDName(name, idNameSpace) 
		 << " Tubs made of " << strucMat[i] << " from 0 to " 
		 << twopi/deg << " with Rin " << rin << " Rout " << rout 
		 << " ZHalf " << dz << endl;
    DDName strucName(DDSplit(strucMat[i]).first, DDSplit(strucMat[i]).second);
    DDMaterial strucMatter(strucName);
    DDLogicalPart strucLogic(DDName(name, idNameSpace), strucMatter, solid);

    DDTranslation r2(0, 0, (dz-diskDz));
    DDpos(DDName(name,idNameSpace), diskName, i+1, r2, DDRotation());
    COUT << "DDTIBRadCableAlgo test: " << DDName(name,idNameSpace) 
		 << " number " << i+1 << " positioned in " << diskName 
		 << " at " << r2 << " with no rotation" << endl;

    //Now the radial cable
    name  = "TIBRadCable" + dbl_to_string(i);
    double rv = layRin[i]+0.5*deltaR;
    vector<double> pgonZ;
    pgonZ.push_back(-0.5*cableT); 
    pgonZ.push_back(cableT*(rv/rMax-0.5));
    pgonZ.push_back(0.5*cableT);
    vector<double> pgonRmin;
    pgonRmin.push_back(rv); 
    pgonRmin.push_back(rv); 
    pgonRmin.push_back(rv); 
    vector<double> pgonRmax;
    pgonRmax.push_back(rMax); 
    pgonRmax.push_back(rMax); 
    pgonRmax.push_back(rv); 
    solid = DDSolidFactory::polycone(DDName(name, idNameSpace), 0, twopi,
				     pgonZ, pgonRmin, pgonRmax);
    COUT << "DDTIBRadCableAlgo test: " << DDName(name, idNameSpace) 
		 << " Polycone made of " << cableMat[i] << " from 0 to "
		 << twopi/deg << " and with " << pgonZ.size() << " sections"
		 << endl;
    for (unsigned int ii = 0; ii <pgonZ.size(); ii++) 
      COUT << "\t" << "\tZ = " << pgonZ[ii] << "\tRmin = " 
		   << pgonRmin[ii] << "\tRmax = " << pgonRmax[ii] << endl;
    DDName cableName(DDSplit(cableMat[i]).first, DDSplit(cableMat[i]).second);
    DDMaterial cableMatter(cableName);
    DDLogicalPart cableLogic(DDName(name, idNameSpace), cableMatter, solid);

    DDTranslation r3(0, 0, (diskDz-(i+0.5)*cableT));
    DDpos(DDName(name,idNameSpace), diskName, i+1, r3, DDRotation());
    COUT << "DDTIBRadCableAlgo test: " << DDName(name,idNameSpace) 
		 << " number " <<i+1 << " positioned in " << diskName << " at "
		 << r3 << " with no rotation" << endl;
    
  }

  //Now the last open zone
  unsigned int i = layRin.size();
  rin  = layRin[i-1]+0.5*(deltaR+cylinderT)+supportDR;
  rout = rMax;
  name  = "TIBOpenZone" + dbl_to_string(i);
  solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
			       rout, 0, twopi);
  COUT << "DDTIBRadCableAlgo test: " << DDName(name, idNameSpace) 
	       << " Tubs made of " << strucMat[i] << " from 0 to " 
	       << twopi/deg << " with Rin " << rin << " Rout " << rout 
	       << " ZHalf " << dz << endl;
  DDName strucName(DDSplit(strucMat[i]).first, DDSplit(strucMat[i]).second);
  DDMaterial strucMatter(strucName);
  DDLogicalPart strucLogic(DDName(name, idNameSpace), strucMatter, solid);

  DDTranslation r2(0, 0, (dz-diskDz));
  DDpos(DDName(name,idNameSpace), diskName, i+1, r2, DDRotation());
  COUT << "DDTIBRadCableAlgo test: " << DDName(name,idNameSpace) 
	       << " number " << i+1 << " positioned in " << diskName 
	       << " at " << r2 << " with no rotation" << endl;

  COUT << "<<== End of DDTIBRadCableAlgo construction ..." << endl;
}
