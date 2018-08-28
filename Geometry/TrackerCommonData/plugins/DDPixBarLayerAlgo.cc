///////////////////////////////////////////////////////////////////////////////
// File: DDPixBarLayerAlgo.cc
// Description: Make one layer of pixel barrel detector
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDPixBarLayerAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDPixBarLayerAlgo::DDPixBarLayerAlgo() {
  LogDebug("PixelGeom") <<"DDPixBarLayerAlgo info: Creating an instance";
}

DDPixBarLayerAlgo::~DDPixBarLayerAlgo() {}

void DDPixBarLayerAlgo::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments & vsArgs) {
  DDCurrentNamespace ns;
  idNameSpace = *ns;
  DDName parentName = parent().name();

  genMat    = sArgs["GeneralMaterial"];
  number    = int(nArgs["Ladders"]);
  layerDz   = nArgs["LayerDz"];
  sensorEdge= nArgs["SensorEdge"];
  coolDz    = nArgs["CoolDz"];
  coolWidth = nArgs["CoolWidth"];
  coolSide  = nArgs["CoolSide"];
  coolThick = nArgs["CoolThick"];
  coolDist  = nArgs["CoolDist"];
  coolMat   = sArgs["CoolMaterial"];
  tubeMat   = sArgs["CoolTubeMaterial"];

  LogDebug("PixelGeom") << "DDPixBarLayerAlgo debug: Parent " << parentName 
			<< " NameSpace " << idNameSpace << "\n"
			<< "\tLadders " << number << "\tGeneral Material " 
			<< genMat << "\tLength " << layerDz << "\tSensorEdge "
			<< sensorEdge << "\tSpecification of Cooling Pieces:\n"
			<< "\tLength " << coolDz << " Width " << coolWidth 
			<< " Side " << coolSide << " Thickness of Shell " 
			<< coolThick << " Radial distance " << coolDist 
			<< " Materials " << coolMat << ", " << tubeMat;

  ladder      = vsArgs["LadderName"];
  ladderWidth = vArgs["LadderWidth"];
  ladderThick = vArgs["LadderThick"];
  
  LogDebug("PixelGeom") << "DDPixBarLayerAlgo debug: Full Ladder " 
			<< ladder[0] << " width/thickness " << ladderWidth[0]
			<< ", " << ladderThick[0] << "\tHalf Ladder " 
			<< ladder[1] << " width/thickness " << ladderWidth[1]
			<< ", " << ladderThick[1];
}

void DDPixBarLayerAlgo::execute(DDCompactView& cpv) {

  DDName      mother = parent().name();
  const std::string &idName = mother.name();

  double dphi = CLHEP::twopi/number;
  double d2   = 0.5*coolWidth;
  double d1   = d2 - coolSide*sin(0.5*dphi);
  double x1   = (d1+d2)/(2.*sin(0.5*dphi));
  double x2   = coolDist*sin(0.5*dphi);
  double rmin = (coolDist-0.5*(d1+d2))*cos(0.5*dphi)-0.5*ladderThick[0];
  double rmax = (coolDist+0.5*(d1+d2))*cos(0.5*dphi)+0.5*ladderThick[0];
  double rmxh = rmax - 0.5*ladderThick[0] + ladderThick[1];
  LogDebug("PixelGeom") << "DDPixBarLayerAlgo test: Rmin/Rmax " << rmin 
			<< ", " << rmax << " d1/d2 " << d1 << ", " << d2 
			<< " x1/x2 " << x1 << ", " << x2;

  double rtmi = rmin + 0.5*ladderThick[0] - ladderThick[1];
  double rtmx = sqrt(rmxh*rmxh+ladderWidth[1]*ladderWidth[1]);
  DDSolid solid = DDSolidFactory::tubs(DDName(idName, idNameSpace),0.5*layerDz,
                                       rtmi, rtmx, 0, CLHEP::twopi);
  LogDebug("PixelGeom") << "DDPixBarLayerAlgo test: " 
			<< DDName(idName, idNameSpace) << " Tubs made of " 
			<< genMat << " from 0 to " << CLHEP::twopi/CLHEP::deg 
			<< " with Rin " << rtmi << " Rout " << rtmx 
			<< " ZHalf " << 0.5*layerDz;
  DDName matname(DDSplit(genMat).first, DDSplit(genMat).second);
  DDMaterial matter(matname);
  DDLogicalPart layer(solid.ddname(), matter, solid);

  double rr = 0.5*(rmax+rmin);
  double dr = 0.5*(rmax-rmin);
  double h1 = 0.5*coolSide*cos(0.5*dphi);
  std::string name = idName + "CoolTube";
  solid = DDSolidFactory::trap(DDName(name,idNameSpace), 0.5*coolDz, 0, 0,
			       h1, d2, d1, 0, h1, d2, d1, 0);
  LogDebug("PixelGeom") << "DDPixBarLayerAlgo test: " <<solid.name() 
			<< " Trap made of " << tubeMat << " of dimensions " 
			<< 0.5*coolDz << ", 0, 0, " << h1 << ", " << d2 
			<< ", " << d1 << ", 0, " << h1 << ", " << d2 << ", " 
			<< d1 << ", 0";
  matter = DDMaterial(DDName(DDSplit(tubeMat).first, DDSplit(tubeMat).second));
  DDLogicalPart coolTube(solid.ddname(), matter, solid);

  name = idName + "Coolant";
  h1  -= coolThick;
  d1  -= coolThick;
  d2  -= coolThick;
  solid = DDSolidFactory::trap(DDName(name,idNameSpace), 0.5*coolDz, 0, 0,
			       h1, d2, d1, 0, h1, d2, d1, 0);
  LogDebug("PixelGeom") << "DDPixBarLayerAlgo test: " <<solid.name() 
			<< " Trap made of " << coolMat << " of dimensions " 
			<< 0.5*coolDz << ", 0, 0, " << h1 << ", " << d2
			<< ", " << d1 << ", 0, " << h1 << ", " << d2 << ", " 
			<< d1 << ", 0";
  matter = DDMaterial(DDName(DDSplit(coolMat).first, DDSplit(coolMat).second));
  DDLogicalPart cool(solid.ddname(), matter, solid);
 cpv.position(cool, coolTube, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  LogDebug("PixelGeom") << "DDPixBarLayerAlgo test: " << cool.name() 
			<< " number 1 positioned in " << coolTube.name() 
			<< " at (0,0,0) with no rotation";

  DDName ladderFull(DDSplit(ladder[0]).first, DDSplit(ladder[0]).second);
  DDName ladderHalf(DDSplit(ladder[1]).first, DDSplit(ladder[1]).second);

  int nphi=number/2, copy=1, iup=-1;
  double phi0 = 90*CLHEP::deg;
  for (int i=0; i<number; i++) {
	
    double phi = phi0 + i*dphi;
    double phix, phiy, rrr, xx;
    std::string rots;
    DDTranslation tran;
    DDRotation rot;
    if (i == 0 || i == nphi) {
      rrr  = rr + dr + 0.5*(ladderThick[1]-ladderThick[0]);
      xx   = (0.5*ladderWidth[1] - sensorEdge) * sin(phi);
      tran = DDTranslation(xx, rrr*sin(phi), 0);
      rots = idName + std::to_string(copy);
      phix = phi-90*CLHEP::deg;
      phiy = 90*CLHEP::deg+phix;
      LogDebug("PixelGeom") << "DDPixBarLayerAlgo test: Creating a new "
			    << "rotation: " << rots << "\t90., " 
			    << phix/CLHEP::deg << ", 90.," << phiy/CLHEP::deg 
			    << ", 0, 0";
      rot = DDrot(DDName(rots,idNameSpace), 90*CLHEP::deg, phix, 90*CLHEP::deg,
		  phiy, 0.,0.);
     cpv.position(ladderHalf, layer, copy, tran, rot);
      LogDebug("PixelGeom") << "DDPixBarLayerAlgo test: " << ladderHalf 
			    << " number " << copy << " positioned in " 
			    << layer.name() << " at " << tran << " with " 
			    << rot;
      copy++;
      iup  = -1;
      rrr  = rr - dr - 0.5*(ladderThick[1]-ladderThick[0]);
      tran = DDTranslation(-xx, rrr*sin(phi), 0);
      rots = idName + std::to_string(copy);
      phix = phi+90*CLHEP::deg;
      phiy = 90*CLHEP::deg+phix;
      LogDebug("PixelGeom") << "DDPixBarLayerAlgo test: Creating a new "
			    << "rotation: " << rots << "\t90., " 
			    << phix/CLHEP::deg << ", 90.," << phiy/CLHEP::deg 
			    << ", 0, 0";
      rot = DDrot(DDName(rots,idNameSpace), 90*CLHEP::deg, phix, 90*CLHEP::deg,
		  phiy, 0.,0.);
     cpv.position(ladderHalf, layer, copy, tran, rot);
      LogDebug("PixelGeom") << "DDPixBarLayerAlgo test: " << ladderHalf 
			    << " number " << copy << " positioned in " 
			    << layer.name() << " at " << tran << " with " 
			    << rot;
      copy++;
    } else {
      iup  =-iup;
      rrr  = rr + iup*dr;
      tran = DDTranslation(rrr*cos(phi), rrr*sin(phi), 0);
      rots = idName + std::to_string(copy);
      if (iup > 0) phix = phi-90*CLHEP::deg;
      else         phix = phi+90*CLHEP::deg;
      phiy = phix+90.*CLHEP::deg;
      LogDebug("PixelGeom") << "DDPixBarLayerAlgo test: Creating a new "
			    << "rotation: " << rots << "\t90., " 
			    << phix/CLHEP::deg << ", 90.," << phiy/CLHEP::deg 
			    << ", 0, 0";
      rot = DDrot(DDName(rots,idNameSpace), 90*CLHEP::deg, phix, 90*CLHEP::deg,
		  phiy, 0.,0.);
     cpv.position(ladderFull, layer, copy, tran, rot);
      LogDebug("PixelGeom") << "DDPixBarLayerAlgo test: " << ladderFull 
			    << " number " << copy << " positioned in " 
			    << layer.name() << " at " << tran << " with " 
			    << rot;
      copy++;
    }
    rrr  = coolDist*cos(0.5*dphi);
    tran = DDTranslation(rrr*cos(phi)-x2*sin(phi), 
			 rrr*sin(phi)+x2*cos(phi), 0);
    rots = idName + std::to_string(i+100);
    phix = phi+0.5*dphi;
    if (iup > 0) phix += 180*CLHEP::deg;
    phiy = phix+90.*CLHEP::deg;
    LogDebug("PixelGeom") << "DDPixBarLayerAlgo test: Creating a new "
			  << "rotation: " << rots << "\t90., " 
			  << phix/CLHEP::deg << ", 90.," << phiy/CLHEP::deg 
			  << ", 0, 0";
    rot = DDrot(DDName(rots,idNameSpace), 90*CLHEP::deg, phix, 90*CLHEP::deg,
		phiy, 0.,0.);
   cpv.position(coolTube, layer, i+1, tran, rot);
    LogDebug("PixelGeom") << "DDPixBarLayerAlgo test: " << coolTube.name() 
			  << " number " << i+1 << " positioned in " 
			  << layer.name() << " at " << tran << " with "<< rot;
  }
}
