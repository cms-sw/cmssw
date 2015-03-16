///////////////////////////////////////////////////////////////////////////////
// File: DDPixBarLayerUpgradeAlgo.cc
// Description: Make one layer of pixel barrel detector for Upgrading.
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "SLHCUpgradeSimulations/Geometry/interface/DDPixBarLayerUpgradeAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDPixBarLayerUpgradeAlgo::DDPixBarLayerUpgradeAlgo() {
  LogDebug("PixelGeom") <<"DDPixBarLayerUpgradeAlgo info: Creating an instance";
}

DDPixBarLayerUpgradeAlgo::~DDPixBarLayerUpgradeAlgo() {}

void DDPixBarLayerUpgradeAlgo::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments & vsArgs) {

  idNameSpace = DDCurrentNamespace::ns();
  DDName parentName = parent().name();

  genMat    = sArgs["GeneralMaterial"];
  number    = int(nArgs["Ladders"]);
  layerDz   = nArgs["LayerDz"];
  coolDz    = nArgs["CoolDz"];
  coolThick = nArgs["CoolThick"];
  coolRadius= nArgs["CoolRadius"];
  coolDist  = nArgs["CoolDist"];
  cool1Offset = nArgs["Cool1Offset"];
  cool2Offset = nArgs["Cool2Offset"];
  coolMat   = sArgs["CoolMaterial"];
  tubeMat   = sArgs["CoolTubeMaterial"];


  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo debug: Parent " << parentName 
			<< " NameSpace " << idNameSpace << "\n"
			<< "\tLadders " << number << "\tGeneral Material " 
			<< genMat << "\tLength " << layerDz << "\tSpecification of Cooling Pieces:\n"
			<< "\tLength " << coolDz << " Thickness of Shell " 
			<< coolThick << " Radial distance " << coolDist 
			<< " Materials " << coolMat << ", " << tubeMat;

  ladder      = sArgs["LadderName"];
  ladderWidth = nArgs["LadderWidth"];
  ladderThick = nArgs["LadderThick"];
  ladderOffset = nArgs["LadderOffset"];
  outerFirst  = int(nArgs["OuterFirst"]);
 
  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo debug: Full Ladder " 
			<< ladder << " width/thickness " << ladderWidth
			<< ", " << ladderThick;
}

void DDPixBarLayerUpgradeAlgo::execute(DDCompactView& cpv) {

  DDName      mother = parent().name();
  std::string idName = DDSplit(mother).first;

  double dphi = CLHEP::twopi/number;
  double x2   = coolDist*sin(0.5*dphi);
  double rtmi = coolDist*cos(0.5*dphi)-(coolRadius+ladderThick);
  double rmxh = coolDist*cos(0.5*dphi)+(coolRadius+ladderThick+ladderOffset);
  double rtmx = sqrt(rmxh*rmxh+ladderWidth*ladderWidth/4);
  DDSolid solid = DDSolidFactory::tubs(DDName(idName, idNameSpace),0.5*layerDz,
                                       rtmi, rtmx, 0, CLHEP::twopi);
  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: " 
			<< DDName(idName, idNameSpace) << " Tubs made of " 
			<< genMat << " from 0 to " << CLHEP::twopi/CLHEP::deg 
			<< " with Rin " << rtmi << " Rout " << rtmx 
			<< " ZHalf " << 0.5*layerDz;
  DDName matname(DDSplit(genMat).first, DDSplit(genMat).second);
  DDMaterial matter(matname);
  DDLogicalPart layer(solid.ddname(), matter, solid);

  double dr = coolRadius+0.5*ladderThick;
  std::string name = idName + "CoolTube";
  solid = DDSolidFactory::tubs(DDName(name,idNameSpace), 0.5*coolDz,
			       0, coolRadius, 0, CLHEP::twopi);
  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: " <<solid.name() 
			<< " Tubs made of " << tubeMat << " from 0 to " <<
			CLHEP::twopi/CLHEP::deg << " with Rout " << coolRadius <<
			" ZHalf " << 0.5*coolDz;		
  matter = DDMaterial(DDName(DDSplit(tubeMat).first, DDSplit(tubeMat).second));
  DDLogicalPart coolTube(solid.ddname(), matter, solid);

  name = idName + "Coolant";
  solid = DDSolidFactory::tubs(DDName(name,idNameSpace), 0.5*coolDz,
			       0, coolRadius-coolThick, 0, CLHEP::twopi);
  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: " <<solid.name() 
			<< " Tubs made of " << tubeMat << " from 0 to " <<
			CLHEP::twopi/CLHEP::deg << " with Rout " << coolRadius-coolThick <<
			" ZHalf " << 0.5*coolDz;		
  matter = DDMaterial(DDName(DDSplit(coolMat).first, DDSplit(coolMat).second));
  DDLogicalPart cool(solid.ddname(), matter, solid);
  cpv.position (cool, coolTube, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: " << cool.name() 
			<< " number 1 positioned in " << coolTube.name() 
			<< " at (0,0,0) with no rotation";

  DDName ladderFull(DDSplit(ladder).first, DDSplit(ladder).second);
  int  copy=1, iup=(-1)*outerFirst;
  int copyoffset=number+2;
  for (int i=1; i<number+1; i++) {
    double phi = i*dphi+90*CLHEP::deg-0.5*dphi; //to start with the interface ladder
    double phix, phiy, rrr, rrroffset;
    std::string rots;
    DDTranslation tran;
    DDRotation rot;
    iup  =-iup;
    if ((i==1)||(i==number/2+1)){
	dr=coolRadius+0.5*ladderThick+ladderOffset; //interface ladder offset
	}else{
	dr=coolRadius+0.5*ladderThick;
	}
    rrr = coolDist*cos(0.5*dphi)+iup*dr;
    tran = DDTranslation(rrr*cos(phi), rrr*sin(phi), 0);
    rots = idName + dbl_to_string(copy);
    if (iup > 0) phix = phi-90*CLHEP::deg;
    else         phix = phi+90*CLHEP::deg;
    phiy = phix+90.*CLHEP::deg;
    LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: Creating a new "
	                  << "rotation: " << rots << "\t90., " << phix/CLHEP::deg 
		          << ", 90.," << phiy/CLHEP::deg << ", 0, 0";
    rot = DDrot(DDName(rots,idNameSpace), 90*CLHEP::deg, phix, 90*CLHEP::deg, phiy, 0.,0.);
    cpv.position (ladderFull, layer, copy, tran, rot);
    LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: " << ladderFull 
	                  << " number " << copy << " positioned in " 
			  << layer.name() << " at " << tran << " with " 
			  << rot;
    copy++;
    rrr  = coolDist*cos(0.5*dphi);
    tran = DDTranslation(rrr*cos(phi)-x2*sin(phi), 
			 rrr*sin(phi)+x2*cos(phi), 0);
    rots = idName + dbl_to_string(i+100);
    phix = phi+0.5*dphi;
    if (iup > 0) phix += 180*CLHEP::deg;
    phiy = phix+90.*CLHEP::deg;
    LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: Creating a new "
			  << "rotation: " << rots << "\t90., " << phix/CLHEP::deg 
			  << ", 90.," << phiy/CLHEP::deg << ", 0, 0";
    rot = DDrot(DDName(rots,idNameSpace), 90*CLHEP::deg, phix, 90*CLHEP::deg, phiy, 0.,0.);
    cpv.position (coolTube, layer, i+1, tran, rot);
    if ((i==1)||(i==number/2+1)){
    	rrroffset = coolDist*cos(0.5*dphi)+iup*ladderOffset;
	tran = DDTranslation(rrroffset*cos(phi)-cool1Offset*sin(phi), 
		rrroffset*sin(phi)+cool1Offset*cos(phi), 0);
    	cpv.position (coolTube, layer, copyoffset, tran, DDRotation());
	copyoffset++;
	tran = DDTranslation(rrroffset*cos(phi)-cool2Offset*sin(phi), 
		rrroffset*sin(phi)+cool2Offset*cos(phi), 0);
    	cpv.position (coolTube, layer, copyoffset, tran, DDRotation());
	copyoffset++;
	} 
   LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: " << coolTube.name() 
			  << " number " << i+1 << " positioned in " 
			  << layer.name() << " at " << tran << " with "<< rot;
  }
}
