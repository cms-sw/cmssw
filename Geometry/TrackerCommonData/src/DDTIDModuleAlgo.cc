///////////////////////////////////////////////////////////////////////////////
// File: DDTIDModuleAlgo.cc
// Description: Creation of a TID Module
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/interface/DDTIDModuleAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTIDModuleAlgo::DDTIDModuleAlgo() {
  LogDebug("TIDGeom") << "DDTIDModuleAlgo info: Creating an instance";
}

DDTIDModuleAlgo::~DDTIDModuleAlgo() {}

void DDTIDModuleAlgo::initialize(const DDNumericArguments & nArgs,
				 const DDVectorArguments & vArgs,
				 const DDMapArguments & ,
				 const DDStringArguments & sArgs,
				 const DDStringVectorArguments & vsArgs) {

  int i;
  genMat       = sArgs["GeneralMaterial"];
  detectorN    = (int)(nArgs["DetectorNumber"]);
  tol          = nArgs["Tolerance"];
  DDName parentName = parent().name(); 

  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug: Parent " << parentName 
		      << " General Material " << genMat 
		      << " Detector Planes " << detectorN
		      << " Tolerance " << tol;

  moduleThick       = nArgs["ModuleThick"];
  detTilt           = nArgs["DetTilt"];
  fullHeight        = nArgs["FullHeight"];
  dlTop             = nArgs["DlTop"];
  dlBottom          = nArgs["DlBottom"];
  dlHybrid          = nArgs["DlHybrid"];

  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug: ModuleThick " 
		      << moduleThick << " Detector Tilt " << detTilt/deg
		      << " Height " << fullHeight << " dl(Top) " << dlTop
		      << " dl(Bottom) " << dlBottom << " dl(Hybrid) "
		      << dlHybrid;

  topFrameName      = sArgs["TopFrameName"];
  topFrameMat       = sArgs["TopFrameMaterial"];
  topFrameHeight    = nArgs["TopFrameHeight"];
  topFrameThick     = nArgs["TopFrameThick"];
  topFrameWidth     = nArgs["TopFrameWidth"];
  bottomFrameHeight = nArgs["BottomFrameHeight"];
  bottomFrameOver   = nArgs["BottomFrameOver"];
  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug: " << topFrameName 
		      << " Material " << topFrameMat << " Height " 
		      << topFrameHeight << " Thickness " << topFrameThick 
		      << " width " << topFrameWidth << " Extra Height at "
		      << "Bottom " << bottomFrameHeight << " Overlap " 
		      << bottomFrameOver;

  sideFrameName     = sArgs["SideFrameName"];
  sideFrameMat      = sArgs["SideFrameMaterial"];
  sideFrameWidth    = nArgs["SideFrameWidth"];
  sideFrameThick    = nArgs["SideFrameThick"];
  sideFrameOver     = nArgs["SideFrameOver"];
  dumyFrameName     = sArgs["DummyFrameName"];
  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug : " << sideFrameName 
		      << " Material " << sideFrameMat << " Width " 
		      << sideFrameWidth << " Thickness " << sideFrameThick
		      << " Overlap " << sideFrameOver << " Dummy "
		      << dumyFrameName;

  waferName         =vsArgs["WaferName"];
  waferMat          = sArgs["WaferMaterial"];
  sideWidth         = nArgs["SideWidth"];
  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug: Wafer Material " 
		      << waferMat << " Side Width " << sideWidth;
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "\twafrName[" << i << "] = " << waferName[i];

  activeName        =vsArgs["ActiveName"];
  activeMat         = sArgs["ActiveMaterial"];
  activeHeight      = nArgs["ActiveHeight"];
  activeThick       = vArgs["ActiveThick"];
  activeRot         = sArgs["ActiveRotation"];
  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug: Active Material " 
		      << activeMat << " Height " << activeHeight 
		      << " rotated by " << activeRot;
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "\tactiveName[" << i << "] = " << activeName[i]
			<< " of thickness " << activeThick[i];

  hybridName        = sArgs["HybridName"];
  hybridMat         = sArgs["HybridMaterial"];
  hybridHeight      = nArgs["HybridHeight"];
  hybridWidth       = nArgs["HybridWidth"];
  hybridThick       = nArgs["HybridThick"];
  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug: " << hybridName 
		      << " Material " << hybridMat << " Height " 
		      << hybridHeight << " Width " << hybridWidth 
		      << " Thickness " << hybridThick;

  pitchName         =vsArgs["PitchName"];
  pitchMat          = sArgs["PitchMaterial"];
  pitchHeight       = nArgs["PitchHeight"];
  pitchThick        = nArgs["PitchThick"];
  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug: Pitch Adapter Material "
		      << pitchMat << " Height " << pitchHeight
		      << " Thickness " << pitchThick;
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") <<  "\tpitchName[" << i << "] = " << pitchName[i];
}

void DDTIDModuleAlgo::execute() {
  
  LogDebug("TIDGeom") << "==>> Constructing DDTIDModuleAlgo...";

  DDName parentName = parent().name(); 
  DDName name;

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
  
  DDSolid    solid = DDSolidFactory::trap(parentName, dz, 0, 0,
					  h1, bl1, bl1, 0, h1, bl2, bl2, 0);
  DDName     matname = DDName(DDSplit(genMat).first, DDSplit(genMat).second);
  DDMaterial matter  = DDMaterial(matname);
  DDLogicalPart module(solid.ddname(), matter, solid);
  LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
		      << " Trap made of " << genMat << " of dimensions " << dz 
		      << ", 0, 0, " << h1  << ", " << bl1 << ", " << bl1 
		      << ", 0, " << h1 << ", " << bl2  << ", " << bl2 
		      << ", 0";

  //Top of the frame
  name    = DDName(DDSplit(topFrameName).first, DDSplit(topFrameName).second);
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
  solid = DDSolidFactory::trap(name, dz, 0, 0, h1, bl1,bl1, 0, h1, bl2,bl2, 0);
  LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
		      << " Trap made of " << matname << " of dimensions "
		      << dz << ", 0, 0, " << h1 << ", " << bl1 << ", " 
		      << bl1 << ", 0, " << h1 << ", " << bl2 << ", " << bl2
		      << ", 0";
  DDLogicalPart topFrame(solid.ddname(), matter, solid);

  //Frame Sides
  name    = DDName(DDSplit(sideFrameName).first,DDSplit(sideFrameName).second);
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
  solid = DDSolidFactory::trap(name, dz, 0, 0, h1, bl1-tol, bl1-tol, 0, h1, 
			       bl2-tol, bl2-tol, 0);
  LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
		      << " Trap made of " << matname << " of dimensions "
		      << dz << ", 0, 0, " << h1 << ", " << bl1-tol << ", " 
		      << bl1-tol << ", 0, " << h1 << ", " << bl2-tol << ", " 
		      << bl2-tol << ", 0";
  DDLogicalPart sideFrame(solid.ddname(), matter, solid);

  name    = DDName(DDSplit(dumyFrameName).first,DDSplit(dumyFrameName).second);
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
  solid = DDSolidFactory::trap(name, dz, 0, 0, h1, bl1,bl1, 0, h1, bl2,bl2, 0);
  LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
		      << " Trap made of " << matname << " of dimensions "
		      << dz << ", 0, 0, " << h1 << ", " << bl1 << ", " 
		      << bl1 << ", 0, " << h1 << ", " << bl2 << ", " << bl2
		      << ", 0";
  DDLogicalPart frame(solid.ddname(), matter, solid);
  DDpos (frame, sideFrame, 1, DDTranslation(0.0, 0.0, zpos), DDRotation());
  LogDebug("TIDGeom") << "DDTIDModuleAlgo test: " << frame.name() 
		      << " number 1 positioned in " << sideFrame.name()
		      << " at (0,0," << zpos << ") with no rotation";

  name    = DDName(DDSplit(hybridName).first, DDSplit(hybridName).second);
  matname = DDName(DDSplit(hybridMat).first, DDSplit(hybridMat).second);
  matter  = DDMaterial(matname);
  double dx = 0.5 * hybridWidth;
  double dy = 0.5 * hybridThick;
  dz        = 0.5 * hybridHeight;
  solid = DDSolidFactory::box(name, dx, dy, dz);
  LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
		      << " Box made of " << matname << " of dimensions " 
		      << dx << ", " << dy << ", " << dz;
  DDLogicalPart hybrid(solid.ddname(), matter, solid);

  // Loop over detectors to be placed
  for (int k = 0; k < detectorN; k++) {

    // Wafer
    name    = DDName(DDSplit(waferName[k]).first,DDSplit(waferName[k]).second);
    matname = DDName(DDSplit(waferMat).first, DDSplit(waferMat).second);
    matter  = DDMaterial(matname);
    if (k == 0 && dlHybrid < dlTop) {
      bl1     = 0.5 * dlTop;
      bl2     = 0.5 * dlBottom;
    } else {
      bl1     = 0.5 * dlBottom;
      bl2     = 0.5 * dlTop;
    }
    h1      = 0.5 * activeThick[k];
    dz      = 0.5 * fullHeight;
    solid = DDSolidFactory::trap(name, dz, 0, 0, h1, bl1, bl1, 0, h1, bl2, bl2,
				 0);
    LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
			<< " Trap made of " << matname << " of dimensions "
			<< dz << ", 0, 0, " << h1 << ", " << bl1 << ", " 
			<< bl1 << ", 0, " << h1 << ", " << bl2 << ", "
			<< bl2 << ", 0";
    DDLogicalPart wafer(solid.ddname(), matter, solid);

    // Active
    name    = DDName(DDSplit(activeName[k]).first,
		     DDSplit(activeName[k]).second);
    matname = DDName(DDSplit(activeMat).first, DDSplit(activeMat).second);
    matter  = DDMaterial(matname);
    bl1    -= sideWidth;
    bl2    -= sideWidth;
    dz      = 0.5 * activeThick[k];
    h1      = 0.5 * activeHeight;
    solid = DDSolidFactory::trap(name, dz, 0, 0, h1, bl2, bl1, 0, h1, bl2, bl1,
				 0);
    LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
			<< " Trap made of " << matname << " of dimensions "
			<< dz << ", 0, 0, " << h1 << ", " << bl2 << ", " 
			<< bl1 << ", 0, " << h1 << ", " << bl2 << ", "
			<< bl1 << ", 0";
    DDLogicalPart active(solid.ddname(), matter, solid);
    std::string rotstr = DDSplit(activeRot).first;
    DDRotation rot;
    if (rotstr != "NULL") {
      std::string rotns = DDSplit(activeRot).second;
      rot               = DDRotation(DDName(rotstr, rotns));
    }
    DDpos (active, wafer, 1, DDTranslation(0.0, 0.0, 0.0), rot);
    LogDebug("TIDGeom") << "DDTIDModuleAlgo test: " << active.name() 
			<< " number 1 positioned in " << wafer.name() 
			<< " at (0, 0, 0) with " << rot;

    //Pitch Adapter
    name    = DDName(DDSplit(pitchName[k]).first,DDSplit(pitchName[k]).second);
    matname = DDName(DDSplit(pitchMat).first, DDSplit(pitchMat).second);
    matter  = DDMaterial(matname);
    if (dlHybrid > dlTop) {
      dz   = 0.5 * dlTop;
    } else {
      dz   = 0.5 * dlBottom;
    }
    if (k == 0) {
      dx      = dz;
      dy      = 0.5 * pitchThick;
      dz      = 0.5 * pitchHeight;
      solid   = DDSolidFactory::box(name, dx, dy, dz);
      LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name()
			  << " Box made of " << matname << " of dimensions"
			  << " " << dx << ", " << dy << ", " << dz;
    } else {
      h1      = 0.5 * pitchThick;
      bl1     = 0.5 * pitchHeight + 0.5 * dz * sin(detTilt);
      bl2     = 0.5 * pitchHeight - 0.5 * dz * sin(detTilt);
      double thet = atan((bl1-bl2)/(2.*dz));
      solid   = DDSolidFactory::trap(name, dz, thet, 0, h1, bl1, bl1, 0, 
				     h1, bl2, bl2, 0);
      LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
			  << " Trap made of " << matname << " of "
			  << "dimensions " << dz << ", " << thet/deg 
			  << ", 0, " << h1 << ", " << bl1 << ", " << bl1 
			  << ", 0, " << h1 << ", " << bl2 << ", " << bl2 
			  << ", 0";
    }
    DDLogicalPart pa(solid.ddname(), matter, solid);
  }

  LogDebug("TIDGeom") << "<<== End of DDTIDModuleAlgo construction ...";
}
