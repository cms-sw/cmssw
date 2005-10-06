#define DEBUG 0
#define COUT if (DEBUG) cout
///////////////////////////////////////////////////////////////////////////////
// File: DDTOBRadCableAlgo.cc
// Description: Equipping the side disks of TOB with cables etc
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "Geometry/TrackerSimData/interface/DDTOBRadCableAlgo.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTOBRadCableAlgo::DDTOBRadCableAlgo():
  rodRin(0),rodRout(0),cableM(0),connM(0),coolR(0),coolM(0),names(0) {
  COUT << "DDTOBRadCableAlgo info: Creating an instance" << endl;
}

DDTOBRadCableAlgo::~DDTOBRadCableAlgo() {}

void DDTOBRadCableAlgo::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & ,
				   const DDStringVectorArguments & vsArgs) {

  idNameSpace  = DDCurrentNamespace::ns();
  unsigned int i;
  DDName parentName = parent().name();
  COUT << "DDTOBRadCableAlgo debug: Parent " << parentName 
		<< " NameSpace " << idNameSpace << endl;

  diskDz       = nArgs["DiskDz"];
  rMax         = nArgs["RMax"];
  cableT       = nArgs["CableT"];     
  rodRin       = vArgs["RodRin"];    
  rodRout      = vArgs["RodRout"];
  cableM       = vsArgs["CableMaterial"];
  COUT << "DDTOBRadCableAlgo debug: Disk Half width " << diskDz 
		<< "\tRMax " << rMax  << "\tCable Thickness " << cableT  
		<< "\tRadii of disk position and cable materials:" << endl;
  for (i=0; i<rodRin.size(); i++)
    COUT << "\t" << i << "\tRin = " << rodRin[i] << "\tRout = " 
		  << rodRout[i] << "  " << cableM[i] << endl;

  connW        = nArgs["ConnW"];     
  connT        = nArgs["ConnT"];    
  connM        = vsArgs["ConnMaterial"];
  COUT << "DDTOBRadCableAlgo debug: Connector Width = " << connW 
		<< "\tThickness = " << connT << "\tMaterials: " << endl;
  for (i=0; i<connM.size(); i++)
    COUT << "\t" << i << "\t" << connM[i] << endl;

  coolW        = nArgs["CoolW"];     
  coolT        = nArgs["CoolT"];    
  coolR        = vArgs["CoolR"];
  coolM        = vsArgs["CoolMaterial"];
  COUT << "DDTOBRadCableAlgo debug: Cool Manifold Width = " << coolW 
		<< "\tThickness = "<<coolT <<"\tRadial position and Materials:"
		<< endl;
  for (i=0; i<coolR.size(); i++)
    COUT << "\t" << i <<"\tR = " << coolR[i] <<" " << coolM[i] <<endl;

  names        = vsArgs["RingName"];      
  COUT << "DDTOBRadCableAlgo debug: Names : ";
  for (i=0; i<names.size(); i++)
    COUT << " " << names[i];
  COUT << endl;
}

void DDTOBRadCableAlgo::execute() {
  
  COUT << "==>> Constructing DDTOBRadCableAlgo..." << endl;
  DDName diskName = parent().name();

  // Loop over sub disks
  for (unsigned int i=0; i<names.size(); i++) {

    DDSolid solid;
    string  name;
    double  dz, rin, rout;
    
    //Cooling Manifold
    name  = "TOBCool" + names[i];
    dz    = 0.5*coolT;
    rin   = coolR[i]-0.5*coolW;
    rout  = coolR[i]+0.5*coolW;
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
				 rout, 0, twopi);
    COUT << "DDTOBRadCableAlgo test: " << DDName(name, idNameSpace) 
		 << " Tubs made of " << coolM[i] << " from 0 to " << twopi/deg 
		 << " with Rin " << rin << " Rout " << rout << " ZHalf " << dz 
		 << endl;
    DDName coolName(DDSplit(coolM[i]).first, DDSplit(coolM[i]).second);
    DDMaterial coolMatter(coolName);
    DDLogicalPart coolLogic(DDName(name, idNameSpace), coolMatter, solid);

    DDTranslation r1(0, 0, (dz-diskDz));
    DDpos(DDName(name,idNameSpace), diskName, i+1, r1, DDRotation());
    COUT << "DDTOBRadCableAlgo test: " << DDName(name,idNameSpace) 
		 << " number " << i+1 << " positioned in " << diskName 
		 << " at " << r1 << " with no rotation" << endl;
    
    //Connectors
    name  = "TOBConn" + names[i];
    dz    = 0.5*connT;
    rin   = 0.5*(rodRin[i]+rodRout[i])-0.5*connW;
    rout  = 0.5*(rodRin[i]+rodRout[i])+0.5*connW;
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
				 rout, 0, twopi);
    COUT << "DDTOBRadCableAlgo test: " << DDName(name, idNameSpace) 
		 << " Tubs made of " << connM[i] << " from 0 to " << twopi/deg 
		 << " with Rin " << rin << " Rout " << rout << " ZHalf " << dz 
		 << endl;
    DDName connName(DDSplit(connM[i]).first, DDSplit(connM[i]).second);
    DDMaterial connMatter(connName);
    DDLogicalPart connLogic(DDName(name, idNameSpace), connMatter, solid);

    DDTranslation r2(0, 0, (dz-diskDz));
    DDpos(DDName(name,idNameSpace), diskName, i+1, r2, DDRotation());
    COUT << "DDTOBRadCableAlgo test: " << DDName(name,idNameSpace) 
		 << " number " << i+1 << " positioned in " << diskName 
		 << " at " << r2 << " with no rotation" << endl;

    //Now the radial cable
    name  = "TOBRadCable" + names[i];
    rin   = 0.5*(rodRin[i]+rodRout[i]);
    vector<double> pgonZ;
    pgonZ.push_back(-0.5*cableT); 
    pgonZ.push_back(cableT*(rin/rMax-0.5));
    pgonZ.push_back(0.5*cableT);
    vector<double> pgonRmin;
    pgonRmin.push_back(rin); 
    pgonRmin.push_back(rin); 
    pgonRmin.push_back(rin); 
    vector<double> pgonRmax;
    pgonRmax.push_back(rMax); 
    pgonRmax.push_back(rMax); 
    pgonRmax.push_back(rin); 
    solid = DDSolidFactory::polycone(DDName(name, idNameSpace), 0, twopi,
				     pgonZ, pgonRmin, pgonRmax);
    COUT << "DDTOBRadCableAlgo test: " << DDName(name, idNameSpace) 
		 << " Polycone made of " << cableM[i] << " from 0 to "
		 << twopi/deg << " and with " << pgonZ.size() << " sections"
		 << endl;
    for (unsigned int ii = 0; ii <pgonZ.size(); ii++) 
      COUT << "\t" << "\tZ = " << pgonZ[ii] << "\tRmin = " 
		   << pgonRmin[ii] << "\tRmax = " << pgonRmax[ii] << endl;
    DDName cableName(DDSplit(cableM[i]).first, DDSplit(cableM[i]).second);
    DDMaterial cableMatter(cableName);
    DDLogicalPart cableLogic(DDName(name, idNameSpace), cableMatter, solid);

    DDTranslation r3(0, 0, (diskDz-(i+0.5)*cableT));
    DDpos(DDName(name,idNameSpace), diskName, i+1, r3, DDRotation());
    COUT << "DDTOBRadCableAlgo test: " << DDName(name,idNameSpace) 
		 << " number " <<i+1 << " positioned in " << diskName << " at "
		 << r3 << " with no rotation" << endl;
    
  }

  COUT << "<<== End of DDTOBRadCableAlgo construction ..." << endl;
}
