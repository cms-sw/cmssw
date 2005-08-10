#define DEBUG 0
#define COUT if (DEBUG) cout
///////////////////////////////////////////////////////////////////////////////
// File: DDPixBarLayerAlgo.cc
// Description: Make one layer of pixel barrel detector
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "Geometry/TrackerSimData/interface/DDPixBarLayerAlgo.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDPixBarLayerAlgo::DDPixBarLayerAlgo() {
  COUT << "DDPixBarLayerAlgo info: Creating an instance" << endl;
}

DDPixBarLayerAlgo::~DDPixBarLayerAlgo() {}

void DDPixBarLayerAlgo::initialize(const DDNumericArguments & nArgs,
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
  coolWidth = nArgs["CoolWidth"];
  coolSide  = nArgs["CoolSide"];
  coolThick = nArgs["CoolThick"];
  coolDist  = nArgs["CoolDist"];
  coolMat   = sArgs["CoolMaterial"];
  tubeMat   = sArgs["CoolTubeMaterial"];

  COUT << "DDPixBarLayerAlgo debug: Parent " << parentName 
		<< " NameSpace " << idNameSpace << endl << "\tLadders "
		<< number << "\tGeneral Material " << genMat << "\tLength "
		<< layerDz << "\tSpecification of Cooling Pieces:" << endl
		<< "\tLength " << coolDz << " Width " << coolWidth << " Side "
		<< coolSide << " Thickness of Shell " << coolThick
		<< " Radial distance " << coolDist << " Materials "
		<< coolMat << ", " << tubeMat << endl;

  ladder      = vsArgs["LadderName"];
  ladderWidth = vArgs["LadderWidth"];
  ladderThick = vArgs["LadderThick"];
  
  COUT << "DDPixBarLayerAlgo debug: Full Ladder " << ladder[0]
		<< " width/thickness " << ladderWidth[0] << ", " 
		<< ladderThick[0] << "\tHalf Ladder " << ladder[1]
		<< " width/thickness " << ladderWidth[1] << ", " 
		<< ladderThick[1] << endl;
}

void DDPixBarLayerAlgo::execute() {

  DDName mother = parent().name();
  string idName = DDSplit(mother).first;

  double dphi = twopi/number;
  double d2   = 0.5*coolWidth;
  double d1   = d2 - coolSide*sin(0.5*dphi);
  double x1   = (d1+d2)/(2.*sin(0.5*dphi));
  double x2   = coolDist*sin(0.5*dphi);
  double rmin = (coolDist-0.5*(d1+d2))*cos(0.5*dphi)-0.5*ladderThick[0];
  double rmax = (coolDist+0.5*(d1+d2))*cos(0.5*dphi)+0.5*ladderThick[0];
  double rmxh = rmax - 0.5*ladderThick[0] + ladderThick[1];
  COUT << "DDPixBarLayerAlgo test: Rmin/Rmax " << rmin << ", " << rmax
	       << " d1/d2 " << d1 << ", " << d2 << " x1/x2 " << x1 << ", "
	       << x2 << endl;

  double rtmi = rmin + 0.5*ladderThick[0] - ladderThick[1];
  double rtmx = sqrt(rmxh*rmxh+ladderWidth[1]*ladderWidth[1]);
  DDSolid solid = DDSolidFactory::tubs(DDName(idName, idNameSpace),0.5*layerDz,
                                       rtmi, rtmx, 0, twopi);
  COUT << "DDPixBarLayerAlgo test: " << DDName(idName, idNameSpace)
               << " Tubs made of " << genMat << " from 0 to " << twopi/deg
               << " with Rin " << rtmi << " Rout " << rtmx << " ZHalf "
               << 0.5*layerDz << endl;
  DDName matname(DDSplit(genMat).first, DDSplit(genMat).second);
  DDMaterial matter(matname);
  DDLogicalPart layer(solid.ddname(), matter, solid);

  double rr = 0.5*(rmax+rmin);
  double dr = 0.5*(rmax-rmin);
  double h1 = 0.5*coolSide*cos(0.5*dphi);
  string name = idName + "CoolTube";
  solid = DDSolidFactory::trap(DDName(name,idNameSpace), 0.5*coolDz, 0, 0,
			       h1, d2, d1, 0, h1, d2, d1, 0);
  COUT << "DDPixBarLayerAlgo test: " <<solid.name() << " Trap made of "
               << tubeMat << " of dimensions " << 0.5*coolDz << ", 0, 0, " 
	       << h1 << ", " << d2 << ", " << d1 << ", 0, " << h1 << ", " 
	       << d2 << ", " << d1 << ", 0" << endl;
  matter = DDMaterial(DDName(DDSplit(tubeMat).first, DDSplit(tubeMat).second));
  DDLogicalPart coolTube(solid.ddname(), matter, solid);

  name = idName + "Coolant";
  h1  -= coolThick;
  d1  -= coolThick;
  d2  -= coolThick;
  solid = DDSolidFactory::trap(DDName(name,idNameSpace), 0.5*coolDz, 0, 0,
			       h1, d2, d1, 0, h1, d2, d1, 0);
  COUT << "DDPixBarLayerAlgo test: " <<solid.name() << " Trap made of "
               << coolMat << " of dimensions " << 0.5*coolDz << ", 0, 0, " 
	       << h1 << ", " << d2 << ", " << d1 << ", 0, " << h1 << ", " 
	       << d2 << ", " << d1 << ", 0" << endl;
  matter = DDMaterial(DDName(DDSplit(coolMat).first, DDSplit(coolMat).second));
  DDLogicalPart cool(solid.ddname(), matter, solid);
  DDpos (cool, coolTube, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  COUT << "DDPixBarLayerAlgo test: " << cool.name()
               << " number 1 positioned in " << coolTube.name()
               << " at (0,0,0) with no rotation" << endl;

  DDName ladderFull(DDSplit(ladder[0]).first, DDSplit(ladder[0]).second);
  DDName ladderHalf(DDSplit(ladder[1]).first, DDSplit(ladder[1]).second);

  int nphi=number/2, copy=1, iup=-1;
  double phi0 = 90*deg;
  for (int i=0; i<number; i++) {
	
    double phi = phi0 + i*dphi;
    double phix, phiy, rrr;
    string rots;
    DDTranslation tran;
    DDRotation rot;
    if (i == 0 || i == nphi) {
      rrr  = rr + dr + 0.5*(ladderThick[1]-ladderThick[0]);
      tran = DDTranslation(0.5*ladderWidth[1]*sin(phi), rrr*sin(phi), 0);
      rots = idName + dbl_to_string(copy);
      phix = phi-90*deg;
      phiy = 90*deg+phix;
      COUT << "DDPixBarLayerAlgo test: Creating a new rotation: "
		   << rots << "\t90., " << phix/deg << ", 90.," << phiy/deg
		   << ", 0, 0" << endl;
      rot = DDrot(DDName(rots,idNameSpace), 90*deg, phix, 90*deg, phiy, 0.,0.);
      DDpos (ladderHalf, layer, copy, tran, rot);
      COUT << "DDPixBarLayerAlgo test: " << ladderHalf << " number " 
		   << copy << " positioned in " << layer.name() << " at " 
		   << tran << " with " << rot << endl;
      copy++;
      iup  = -1;
      rrr  = rr - dr - 0.5*(ladderThick[1]-ladderThick[0]);
      tran = DDTranslation(-0.5*ladderWidth[1]*sin(phi), rrr*sin(phi), 0);
      rots = idName + dbl_to_string(copy);
      phix = phi+90*deg;
      phiy = 90*deg+phix;
      COUT << "DDPixBarLayerAlgo test: Creating a new rotation: "
		   << rots << "\t90., " << phix/deg << ", 90.," << phiy/deg
		   << ", 0, 0" << endl;
      rot = DDrot(DDName(rots,idNameSpace), 90*deg, phix, 90*deg, phiy, 0.,0.);
      DDpos (ladderHalf, layer, copy, tran, rot);
      COUT << "DDPixBarLayerAlgo test: " << ladderHalf << " number " 
		   << copy << " positioned in " << layer.name() << " at " 
		   << tran << " with " << rot << endl;
      copy++;
    } else {
      iup  =-iup;
      rrr  = rr + iup*dr;
      tran = DDTranslation(rrr*cos(phi), rrr*sin(phi), 0);
      rots = idName + dbl_to_string(copy);
      if (iup > 0) phix = phi-90*deg;
      else         phix = phi+90*deg;
      phiy = phix+90.*deg;
      COUT << "DDPixBarLayerAlgo test: Creating a new rotation: "
		   << rots << "\t90., " << phix/deg << ", 90.," << phiy/deg
		   << ", 0, 0" << endl;
      rot = DDrot(DDName(rots,idNameSpace), 90*deg, phix, 90*deg, phiy, 0.,0.);
      DDpos (ladderFull, layer, copy, tran, rot);
      COUT << "DDPixBarLayerAlgo test: " << ladderFull << " number " 
		   << copy << " positioned in " << layer.name() << " at " 
		   << tran << " with " << rot << endl;
      copy++;
    }
    rrr  = coolDist*cos(0.5*dphi);
    tran = DDTranslation(rrr*cos(phi)-x2*sin(phi), 
			 rrr*sin(phi)+x2*cos(phi), 0);
    rots = idName + dbl_to_string(i+100);
    phix = phi+0.5*dphi;
    if (iup > 0) phix += 180*deg;
    phiy = phix+90.*deg;
    COUT << "DDPixBarLayerAlgo test: Creating a new rotation: "
		 << rots << "\t90., " << phix/deg << ", 90.," << phiy/deg
		 << ", 0, 0" << endl;
    rot = DDrot(DDName(rots,idNameSpace), 90*deg, phix, 90*deg, phiy, 0.,0.);
    DDpos (coolTube, layer, i+1, tran, rot);
    COUT << "DDPixBarLayerAlgo test: " << coolTube.name() << " number "
		 << i+1 << " positioned in " << layer.name() << " at " << tran 
		 << " with " << rot << endl;
  }
}
