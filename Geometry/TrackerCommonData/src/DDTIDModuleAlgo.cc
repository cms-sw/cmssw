#define DEBUG 0
#define COUT if (DEBUG) cout
///////////////////////////////////////////////////////////////////////////////
// File: DDTIDModuleAlgo.cc
// Description: Creation of a TID Module
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "Geometry/TrackerSimData/interface/DDTIDModuleAlgo.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Base/interface/DDTypes.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTIDModuleAlgo::DDTIDModuleAlgo():
  topFrameZ(0),sideFrameZ(0),waferRot(0),activeThick(0),activeZ(0),hybridZ(0),
  pitchZ(0) {
  COUT << "DDTIDModuleAlgo info: Creating an instance" << endl;
}

DDTIDModuleAlgo::~DDTIDModuleAlgo() {}

void DDTIDModuleAlgo::initialize(const DDNumericArguments & nArgs,
				 const DDVectorArguments & vArgs,
				 const DDMapArguments & ,
				 const DDStringArguments & sArgs,
				 const DDStringVectorArguments & vsArgs) {

  idNameSpace  = DDCurrentNamespace::ns();
  genMat       = sArgs["GeneralMaterial"];

  unsigned int i;
  DDName parentName = parent().name(); 

  COUT << "DDTIDModuleAlgo debug: Parent " << parentName 
		<< " NameSpace " << idNameSpace << " General Material " 
		<< genMat << endl;

  moduleThick       = nArgs["ModuleThick"];
  detTilt           = nArgs["DetTilt"];
  fullHeight        = nArgs["FullHeight"];
  dlTop             = nArgs["DlTop"];
  dlBottom          = nArgs["DlBottom"];
  dlHybrid          = nArgs["DlHybrid"];

  COUT << "DDTIDModuleAlgo debug: ModuleThick " << moduleThick 
		<< " Detector Tilt " << detTilt/deg << " Height " << fullHeight
		<< " dl(Top) " << dlTop << " dl(Bottom) " << dlBottom 
		<< " dl(Hybrid) " << dlHybrid << endl;

  topFrameMat       = sArgs["TopFrameMaterial"];
  topFrameHeight    = nArgs["TopFrameHeight"];
  topFrameThick     = nArgs["TopFrameThick"];
  topFrameWidth     = nArgs["TopFrameWidth"];
  topFrameZ         = vArgs["TopFrameZ"];
  bottomFrameHeight = nArgs["BottomFrameHeight"];
  bottomFrameOver   = nArgs["BottomFrameOver"];
  COUT << "DDTIDModuleAlgo debug: Top Frame Material " << topFrameMat 
		<< " Height " << topFrameHeight << " Thickness " 
		<< topFrameThick << " width " << topFrameWidth 
		<< " positioned at";
  for (i = 0; i < topFrameZ.size(); i++)
    COUT << " " << topFrameZ[i];
  COUT << " Extra Height at Bottom " << bottomFrameHeight
		<< " Overlap " <<bottomFrameOver << endl;

  sideFrameMat      = sArgs["SideFrameMaterial"];
  sideFrameWidth    = nArgs["SideFrameWidth"];
  sideFrameThick    = nArgs["SideFrameThick"];
  sideFrameOver     = nArgs["SideFrameOver"];
  sideFrameZ        = vArgs["SideFrameZ"];
  COUT << "DDTIDModuleAlgo debug : Side Frame Material " 
		<< sideFrameMat << " Width " << sideFrameWidth << " Thickness "
		<< sideFrameThick  << " Overlap " << sideFrameOver 
		<< " positioned at";
  for (i = 0; i < sideFrameZ.size(); i++)
    COUT << " " << sideFrameZ[i];
  COUT << " Extra Height at Bottom " << bottomFrameHeight
		<< " Overlap " <<bottomFrameOver << endl;

  waferMat          = sArgs["WaferMaterial"];
  sideWidth         = nArgs["SideWidth"];
  waferRot          =vsArgs["WaferRotation"];
  COUT << "DDTIDModuleAlgo debug: Wafer Material " << waferMat 
		<< " Side Width " << sideWidth << " positioned with rotation"
		<< " matrix:";
  for (i=0; i<waferRot.size(); i++)
    COUT << " " << waferRot[i];
  COUT << endl;

  activeMat         = sArgs["ActiveMaterial"];
  activeHeight      = nArgs["ActiveHeight"];
  activeThick       = vArgs["ActiveThick"];
  activeRot         = sArgs["ActiveRotation"];
  activeZ           = vArgs["ActiveZ"];
  COUT << "DDTIDModuleAlgo debug: Active Material " << activeMat 
		<< " Height " << activeHeight << " rotated by " << activeRot 
		<< " Thickness/Z";
  for (i=0; i<activeThick.size(); i++)
    COUT << " " << activeThick[i] << " " << activeZ[i];
  COUT << endl;

  hybridMat         = sArgs["HybridMaterial"];
  hybridHeight      = nArgs["HybridHeight"];
  hybridWidth       = nArgs["HybridWidth"];
  hybridThick       = nArgs["HybridThick"];
  hybridZ           = vArgs["HybridZ"];
  COUT << "DDTIDModuleAlgo debug: Hybrid Material " << hybridMat 
		<< " Height " << hybridHeight << " Width " << hybridWidth 
		<< " Thickness " << hybridThick << " Z";
  for (i=0; i<hybridZ.size(); i++)
    COUT << " " << hybridZ[i];
  COUT << endl;

  pitchMat          = sArgs["PitchMaterial"];
  pitchHeight       = nArgs["PitchHeight"];
  pitchThick        = nArgs["PitchThick"];
  pitchZ            = vArgs["PitchZ"];
  pitchRot          = sArgs["PitchRotation"];
  COUT << "DDTIDModuleAlgo debug: Pitch Adapter Material " << pitchMat
		<< " Height " << pitchHeight << " Thickness " << pitchThick 
		<< " position at Z";
  for (i=0; i<pitchZ.size(); i++)
    COUT << " " << pitchZ[i];
  COUT << " with rotation " << pitchRot << endl;
}

void DDTIDModuleAlgo::execute() {
  
  COUT << "==>> Constructing DDTIDModuleAlgo..." << endl;

  DDName parentName = parent().name(); 
  string idName = DDSplit(parentName).first;
  DDName matname(DDSplit(genMat).first, DDSplit(genMat).second);
  DDMaterial matter(matname);
  double dzdif = fullHeight + topFrameHeight;
  double topfr = bottomFrameHeight - bottomFrameOver;
  double dxbot, dxtop;
  if (dlHybrid > dlTop) {
    dxbot = 0.5*dlBottom + sideFrameWidth - sideFrameOver;
    dxtop = 0.5*dlHybrid + sideFrameWidth - sideFrameOver;
    dxbot = dxtop - (dxtop-dxbot)*(topfr+dzdif)/dzdif;
  } else {
    dxbot = 0.5*dlHybrid + sideFrameWidth - sideFrameOver;
    dxtop = 0.5*dlTop    + sideFrameWidth - sideFrameOver;
    dxtop = dxbot + (dxtop-dxbot)*(topfr+dzdif)/dzdif;
  }
  double dxdif = dxtop - dxbot;
  double bl1   = dxbot;
  double bl2   = dxtop;
  double h1    = 0.5 * moduleThick;
  double dz    = 0.5 * (dzdif + topfr);
  
  DDSolid solid = DDSolidFactory::trap(DDName(idName,idNameSpace), dz, 0, 0,
				       h1, bl1, bl1, 0, h1, bl2, bl2, 0);
  COUT << "DDTIDModuleAlgo test:\t" << solid.name() << " Trap made of "
	       << matname << " of dimensions " << dz << ", 0, 0, " << h1 
	       << ", " << bl1 << ", " << bl1 << ", 0, " << h1 << ", " << bl2 
	       << ", " << bl2 << ", 0" << endl;
  DDLogicalPart module(solid.ddname(), matter, solid);

  //Top of the frame
  string name = idName + "TopFrame";
  matname = DDName(DDSplit(topFrameMat).first, DDSplit(topFrameMat).second);
  matter  = DDMaterial(matname);
  if (dlHybrid > dlTop) {
    bl1 = 0.5 * (dlTop + topFrameWidth);
    bl2 = bl1 + topFrameHeight*dxdif/(dzdif+topfr);
  } else {
    bl2 = 0.5 * (dlBottom + topFrameWidth);
    bl1 = bl2 - topFrameHeight*dxdif/(dzdif+topfr);
  }
  h1 = 0.5 * topFrameThick;
  dz = 0.5 * topFrameHeight;
  solid = DDSolidFactory::trap(DDName(name,idNameSpace), dz, 0, 0, h1, bl1, 
			       bl1, 0, h1, bl2, bl2, 0);
  COUT << "DDTIDModuleAlgo test:\t" << solid.name() << " Trap made of "
	       << matname << " of dimensions " << dz << ", 0, 0, " << h1 
	       << ", " << bl1 << ", "  << bl1 << ", 0, " << h1 << ", " << bl2 
	       << ", " << bl2 << ", 0" << endl;
  DDLogicalPart topFrame(solid.ddname(), matter, solid);

  //Frame Sides
  name    = idName + "SideFrame";
  matname = DDName(DDSplit(sideFrameMat).first, DDSplit(sideFrameMat).second);
  matter  = DDMaterial(matname);
  if (dlHybrid > dlTop) {
    bl2 = 0.5*dlTop    + sideFrameWidth - sideFrameOver;
    bl1 = dxbot;
  } else {
    bl1 = 0.5*dlBottom + sideFrameWidth - sideFrameOver;
    bl2 = dxtop;
  }
  h1 = 0.5 * sideFrameThick;
  dz = 0.5 * (fullHeight + topfr);
  solid = DDSolidFactory::trap(DDName(name,idNameSpace), dz, 0, 0, h1, bl1, 
			       bl1, 0, h1, bl2, bl2, 0);
  COUT << "DDTIDModuleAlgo test:\t" << solid.name() << " Trap made of "
	       << matname << " of dimensions " << dz << ", 0, 0, " << h1 
	       << ", " << bl1 << ", " << bl1 << ", 0, " << h1 << ", " << bl2 
	       << ", " << bl2 << ", 0" << endl;
  DDLogicalPart sideFrame(solid.ddname(), matter, solid);

  name    = idName + "Frame";
  matname = DDName(DDSplit(genMat).first, DDSplit(genMat).second);
  matter  = DDMaterial(matname);
  double zpos;
  dz      = fullHeight - bottomFrameOver;
  if (dlHybrid > dlTop) {
    bl2    -= sideFrameWidth;
    bl1     = bl2 - dz*dxdif/(dzdif+topfr);
    zpos    = 0.5 * (fullHeight + topfr - dz);
  } else {
    bl1    -= sideFrameWidth;
    bl2     = bl1 + dz*dxdif/(dzdif+topfr);
    zpos    =-0.5 * (fullHeight + topfr - dz);
  }
  dz     /= 2.;
  solid = DDSolidFactory::trap(DDName(name,idNameSpace), dz, 0, 0, h1, bl1, 
			       bl1, 0, h1, bl2, bl2, 0);
  COUT << "DDTIDModuleAlgo test:\t" << solid.name() << " Trap made of "
	       << matname << " of dimensions " << dz << ", 0, 0, " << h1 
	       << ", " << bl1 << ", " << bl1 << ", 0, " << h1 << ", " << bl2 
	       << ", " << bl2 << ", 0" << endl;
  DDLogicalPart frame(solid.ddname(), matter, solid);
  DDpos (frame, sideFrame, 1, DDTranslation(0.0, 0.0, zpos), DDRotation());
  COUT << "DDTIDModuleAlgo test: " << frame.name() 
	       << " number 1 positioned in " << sideFrame.name()
	       << " at (0,0," << zpos << ") with no rotation" << endl;

  name    = idName + "Hybrid";
  matname = DDName(DDSplit(hybridMat).first, DDSplit(hybridMat).second);
  matter  = DDMaterial(matname);
  double dx = 0.5 * hybridWidth;
  double dy = 0.5 * hybridThick;
  dz        = 0.5 * hybridHeight;
  solid = DDSolidFactory::box(DDName(name, idNameSpace), dx, dy, dz);
  COUT << "DDTIDModuleAlgo test:\t" << solid.name() << " Box made of " 
	       << matname << " of dimensions " << dx << ", " << dy << ", " 
	       << dz << endl;
  DDLogicalPart hybrid(solid.ddname(), matter, solid);

  // Loop over detectors to be placed
  for (unsigned int k = 0; k < waferRot.size(); k++) {
    string tag("Rphi");
    if (k>0) tag = "Stereo";

    // Wafer
    name    = idName + tag + "Wafer";
    matname = DDName(DDSplit(waferMat).first, DDSplit(waferMat).second);
    matter  = DDMaterial(matname);
    bl1     = 0.5 * dlBottom;
    bl2     = 0.5 * dlTop;
    h1      = 0.5 * activeThick[k];
    dz      = 0.5 * fullHeight;
    solid = DDSolidFactory::trap(DDName(name,idNameSpace), dz, 0, 0, h1, bl1, 
				 bl1, 0, h1, bl2, bl2, 0);
    COUT << "DDTIDModuleAlgo test:\t" << solid.name() 
		 << " Trap made of " << matname << " of dimensions " << dz 
		 << ", 0, 0, " << h1 << ", " << bl1 << ", " << bl1 << ", 0, " 
		 << h1 << ", " << bl2 << ", " << bl2 << ", 0" << endl;
    DDLogicalPart wafer(solid.ddname(), matter, solid);
    double zpos;
    double ypos = activeZ[k];
    if (dlHybrid > dlTop) {
      zpos =-0.5*(topFrameHeight-topfr);
    } else {
      zpos = 0.5*(topFrameHeight-topfr);
    }
    DDTranslation tran(0, ypos, zpos);
    string rotstr = DDSplit(waferRot[k]).first;
    string rotns;
    DDRotation rot;
    if (rotstr != "NULL") {
      rotns = DDSplit(waferRot[k]).second;
      rot   = DDRotation(DDName(rotstr, rotns));
    }
    DDpos (wafer, module, k+1, tran, rot);
    COUT << "DDTIDModuleAlgo test: " << wafer.name() << " number " 
		 << k+1 << " positioned in " << module.name() << " at " 
		 << tran << " with " << rot << endl;

    // Active
    name    = idName + tag + "Active";
    matname = DDName(DDSplit(activeMat).first, DDSplit(activeMat).second);
    matter  = DDMaterial(matname);
    bl1    -= sideWidth;
    bl2    -= sideWidth;
    dz      = 0.5 * activeThick[k];
    h1      = 0.5 * activeHeight;
    solid = DDSolidFactory::trap(DDName(name,idNameSpace), dz, 0, 0, h1, bl1, 
				 bl2, 0, h1, bl1, bl2, 0);
    COUT << "DDTIDModuleAlgo test:\t" << solid.name() 
		 << " Trap made of " << matname << " of dimensions " << dz 
		 << ", 0, 0, " << h1 << ", " << bl1 << ", "  << bl2 << ", 0, " 
		 << h1 << ", " << bl1 << ", " << bl2 << ", 0" << endl;
    DDLogicalPart active(solid.ddname(), matter, solid);
    rotstr = DDSplit(activeRot).first;
    rot    = DDRotation();
    if (rotstr != "NULL") {
      rotns = DDSplit(activeRot).second;
      rot   = DDRotation(DDName(rotstr, rotns));
    }
    DDpos (active, wafer, 1, DDTranslation(0.0, 0.0, 0.0), rot);
    COUT << "DDTIDModuleAlgo test: " << active.name() 
		 << " number 1 positioned in " << wafer.name() 
		 << " at (0, 0, 0) with " << rot << endl;

    //Pitch Adapter
    name    = idName + tag + "PA";
    matname = DDName(DDSplit(pitchMat).first, DDSplit(pitchMat).second);
    matter  = DDMaterial(matname);
    if (dlHybrid > dlTop) {
      dz   = 0.5 * dlTop;
      zpos = 0.5 * (dzdif+topfr-pitchHeight) - hybridHeight;
    } else {
      dz   = 0.5 * dlBottom;
      zpos =-0.5 * (dzdif+topfr-pitchHeight) + hybridHeight;
    }
    ypos = pitchZ[k];
    double xpos = 0;
    if (k == 0) {
      dx      = dz;
      dy      = 0.5 * pitchThick;
      dz      = 0.5 * pitchHeight;
      solid   = DDSolidFactory::box(DDName(name, idNameSpace), dx, dy, dz);
      COUT << "DDTIDModuleAlgo test:\t" << solid.name() 
		   << " Box made of " << matname << " of dimensions " << dx 
		   << ", " << dy << ", " << dz << endl;
      rot     = DDRotation();
    } else {
      h1      = 0.5 * pitchThick;
      bl1     = 0.5 * pitchHeight + 0.5 * dz * sin(detTilt);
      bl2     = 0.5 * pitchHeight - 0.5 * dz * sin(detTilt);
      double thet = atan((bl1-bl2)/(2.*dz));
      solid   = DDSolidFactory::trap(DDName(name,idNameSpace), dz, thet, 0, h1,
				     bl1, bl1, 0, h1, bl2, bl2, 0);
      COUT << "DDTIDModuleAlgo test:\t" << solid.name() 
		   << " Trap made of " << matname << " of dimensions " << dz 
		   << ", " << thet/deg << ", 0, " << h1 << ", " << bl1 << ", " 
		   << bl1 << ", 0, " << h1 << ", " << bl2 << ", " << bl2 
		   << ", 0" <<endl;
      xpos    = 0.5 * fullHeight * sin(detTilt);
      rotstr  = DDSplit(pitchRot).first;
      rotns   = DDSplit(pitchRot).second;
      rot     = DDRotation(DDName(rotstr, rotns));
    }
    DDLogicalPart pa(solid.ddname(), matter, solid);
    tran = DDTranslation(xpos,ypos,zpos);
    DDpos (pa, module, k+1, tran, rot);
    COUT << "DDTIDModuleAlgo test: " << pa.name() << " number " << k+1 
		 << " positioned in " << module.name() << " at " << tran 
		 << " with " << rot << endl;

    // Position the hybrid now
    ypos = hybridZ[k];
    if (dlHybrid > dlTop) {
      zpos = 0.5 * (dzdif+topfr-hybridHeight);
    } else {
      zpos =-0.5 * (dzdif+topfr-hybridHeight);
    }
    tran = DDTranslation(0,ypos,zpos);
    rot  = DDRotation();
    DDpos (hybrid, module, k+1, tran, rot);
    COUT << "DDTIDModuleAlgo test: " << hybrid.name() << " number " 
		 << k+1 << " positioned in " << module.name() << " at " << tran
		 << " with " << rot << endl;

    // Position the frame
    ypos = topFrameZ[k];
    if (dlHybrid > dlTop) {
      zpos = 0.5 * (dzdif+topfr-topFrameHeight);
    } else {
      zpos =-0.5 * (dzdif+topfr-topFrameHeight);
    }
    tran = DDTranslation(0,ypos,zpos);
    rot  = DDRotation();
    DDpos (topFrame, module, k+1, tran, rot);
    COUT << "DDTIDModuleAlgo test: " << topFrame.name() << " number " 
		 << k+1 << " positioned in " << module.name() << " at " << tran
		 << " with " << rot << endl;

    ypos = sideFrameZ[k];
    if (dlHybrid > dlTop) {
      zpos =-0.5 * topFrameHeight;
    } else {
      zpos = 0.5 * topFrameHeight;
    }
    tran = DDTranslation(0,ypos,zpos);
    rot  = DDRotation();
    DDpos (sideFrame, module, k+1, tran, rot);
    COUT << "DDTIDModuleAlgo test: " << sideFrame.name() << " number " 
		 << k+1 << " positioned in " << module.name() << " at " << tran
		 << " with " << rot << endl;
  }

  COUT << "<<== End of DDTIDModuleAlgo construction ..." << endl;
}
