///////////////////////////////////////////////////////////////////////////////
// File: DDTOBRadCableAlgo.cc
// Description: Equipping the side disks of TOB with cables etc
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
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
  rodRin(0),rodRout(0),cableM(0),connM(0),coolR(0),coolM(0),names(0) {
  edm::LogInfo("TrackerGeom") <<"DDTOBRadCableAlgo info: Creating an instance";
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
  LogDebug("TrackerGeom") << "DDTOBRadCableAlgo debug: Parent " << parentName
			  << " NameSpace " << idNameSpace;

  diskDz       = nArgs["DiskDz"];
  rMax         = nArgs["RMax"];
  cableT       = nArgs["CableT"];     
  rodRin       = vArgs["RodRin"];    
  rodRout      = vArgs["RodRout"];
  cableM       = vsArgs["CableMaterial"];
  LogDebug("TrackerGeom") << "DDTOBRadCableAlgo debug: Disk Half width " 
			  << diskDz << "\tRMax " << rMax  
			  << "\tCable Thickness " << cableT 
			  << "\tRadii of disk position and cable materials:";
  for (i=0; i<rodRin.size(); i++)
    LogDebug("TrackerGeom") << "\t[" << i << "]\tRin = " << rodRin[i] 
			    << "\tRout = " << rodRout[i] << "  " << cableM[i];

  connW        = nArgs["ConnW"];     
  connT        = nArgs["ConnT"];    
  connM        = vsArgs["ConnMaterial"];
  LogDebug("TrackerGeom") << "DDTOBRadCableAlgo debug: Connector Width = " 
			  << connW << "\tThickness = " << connT 
			  << "\tMaterials: ";
  for (i=0; i<connM.size(); i++)
    LogDebug("TrackerGeom") << "\tconnM[" << i << "] = " << connM[i];

  coolW        = nArgs["CoolW"];     
  coolT        = nArgs["CoolT"];    
  coolR        = vArgs["CoolR"];
  coolM        = vsArgs["CoolMaterial"];
  LogDebug("TrackerGeom") << "DDTOBRadCableAlgo debug: Cool Manifold Width = "
			  << coolW << "\tThickness = "<<coolT 
			  <<"\tRadial position and Materials:";
  for (i=0; i<coolR.size(); i++)
    LogDebug("TrackerGeom") << "\t[" << i <<"]\tR = " << coolR[i] 
			    << "\tMaterial = " << coolM[i];

  names        = vsArgs["RingName"];      
  for (i=0; i<names.size(); i++)
    LogDebug("TrackerGeom") << "DDTOBRadCableAlgo debug: names[" << i
			    << "] = " << names[i];
}

void DDTOBRadCableAlgo::execute() {
  
  LogDebug("TrackerGeom") << "==>> Constructing DDTOBRadCableAlgo...";
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
    LogDebug("TrackerGeom") << "DDTOBRadCableAlgo test: " 
			    << DDName(name, idNameSpace) << " Tubs made of " 
			    << coolM[i] << " from 0 to " << twopi/deg 
			    << " with Rin " << rin << " Rout " << rout
			    << " ZHalf " << dz;
    DDName coolName(DDSplit(coolM[i]).first, DDSplit(coolM[i]).second);
    DDMaterial coolMatter(coolName);
    DDLogicalPart coolLogic(DDName(name, idNameSpace), coolMatter, solid);

    DDTranslation r1(0, 0, (dz-diskDz));
    DDpos(DDName(name,idNameSpace), diskName, i+1, r1, DDRotation());
    LogDebug("TrackerGeom") << "DDTOBRadCableAlgo test: " 
			    << DDName(name,idNameSpace) << " number " << i+1 
			    << " positioned in " << diskName << " at " << r1
			    << " with no rotation";
    
    //Connectors
    name  = "TOBConn" + names[i];
    dz    = 0.5*connT;
    rin   = 0.5*(rodRin[i]+rodRout[i])-0.5*connW;
    rout  = 0.5*(rodRin[i]+rodRout[i])+0.5*connW;
    solid = DDSolidFactory::tubs(DDName(name, idNameSpace), dz, rin, 
				 rout, 0, twopi);
    LogDebug("TrackerGeom") << "DDTOBRadCableAlgo test: " 
			    << DDName(name, idNameSpace) << " Tubs made of " 
			    << connM[i] << " from 0 to " << twopi/deg
			    << " with Rin " << rin << " Rout " << rout 
			    << " ZHalf " << dz;
    DDName connName(DDSplit(connM[i]).first, DDSplit(connM[i]).second);
    DDMaterial connMatter(connName);
    DDLogicalPart connLogic(DDName(name, idNameSpace), connMatter, solid);

    DDTranslation r2(0, 0, (dz-diskDz));
    DDpos(DDName(name,idNameSpace), diskName, i+1, r2, DDRotation());
    LogDebug("TrackerGeom") << "DDTOBRadCableAlgo test: " 
			    << DDName(name,idNameSpace) << " number " << i+1 
			    << " positioned in " << diskName << " at " << r2 
			    << " with no rotation";

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
    LogDebug("TrackerGeom") << "DDTOBRadCableAlgo test: " 
			    << DDName(name, idNameSpace) <<" Polycone made of "
			    << cableM[i] << " from 0 to " << twopi/deg
			    << " and with " << pgonZ.size() << " sections";
    for (unsigned int ii = 0; ii <pgonZ.size(); ii++) 
      LogDebug("TrackerGeom") << "\t[" << ii << "]\tZ = " << pgonZ[ii] 
			      << "\tRmin = " << pgonRmin[ii] << "\tRmax = " 
			      << pgonRmax[ii];
    DDName cableName(DDSplit(cableM[i]).first, DDSplit(cableM[i]).second);
    DDMaterial cableMatter(cableName);
    DDLogicalPart cableLogic(DDName(name, idNameSpace), cableMatter, solid);

    DDTranslation r3(0, 0, (diskDz-(i+0.5)*cableT));
    DDpos(DDName(name,idNameSpace), diskName, i+1, r3, DDRotation());
    LogDebug("TrackerGeom") << "DDTOBRadCableAlgo test: " 
			    << DDName(name,idNameSpace) << " number " <<i+1
			    << " positioned in " << diskName << " at " << r3
			    << " with no rotation";
    
  }

  LogDebug("TrackerGeom") << "<<== End of DDTOBRadCableAlgo construction ...";
}
