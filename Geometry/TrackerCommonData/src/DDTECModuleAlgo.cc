///////////////////////////////////////////////////////////////////////////////
// File: DDTECModuleAlgo.cc
// Description: Creation of a TEC Module
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
#include "Geometry/TrackerCommonData/interface/DDTECModuleAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTECModuleAlgo::DDTECModuleAlgo():
  topFrameZ(0),sideFrameZ(0),waferRot(0),activeThick(0),activeZ(0),hybridZ(0),
  pitchZ(0) {
  edm::LogInfo("TECGeom") << "DDTECModuleAlgo info: Creating an instance";
}

DDTECModuleAlgo::~DDTECModuleAlgo() {}

void DDTECModuleAlgo::initialize(const DDNumericArguments & nArgs,
				 const DDVectorArguments & vArgs,
				 const DDMapArguments & ,
				 const DDStringArguments & sArgs,
				 const DDStringVectorArguments & vsArgs) {

  idNameSpace  = DDCurrentNamespace::ns();
  genMat       = sArgs["GeneralMaterial"];

  DDName parentName = parent().name(); 

  LogDebug("TECGeom") << "DDTECModuleAlgo debug: Parent " << parentName 
		      << " NameSpace " << idNameSpace << " General Material "
		      << genMat;

  moduleThick    = nArgs["ModuleThick"];
  detTilt        = nArgs["DetTilt"];
  fullHeight     = nArgs["FullHeight"];
  dlTop          = nArgs["DlTop"];
  dlBottom       = nArgs["DlBottom"];
  dlHybrid       = nArgs["DlHybrid"];

  LogDebug("TECGeom") << "DDTECModuleAlgo debug: ModuleThick " << moduleThick
		      << " Detector Tilt " << detTilt/deg << " Height "
		      << fullHeight << " dl(Top) " << dlTop << " dl(Bottom) "
		      << dlBottom << " dl(Hybrid) " << dlHybrid;

  frameWidth     = nArgs["FrameWidth"];
  frameThick     = nArgs["FrameThick"];
  frameOver      = nArgs["FrameOver"];
  LogDebug("TECGeom") << "DDTECModuleAlgo debug: Frame Width " << frameWidth 
		      << " Thickness " << frameThick << " Overlap " 
		      << frameOver;

  topFrameMat    = sArgs["TopFrameMaterial"];
  topFrameHeight = nArgs["TopFrameHeight"];
  topFrameThick  = nArgs["TopFrameThick"];
  topFrameZ      = vArgs["TopFrameZ"];
  LogDebug("TECGeom") << "DDTECModuleAlgo debug: Top Frame Material " 
		      << topFrameMat << " Height " << topFrameHeight 
		      << " Thickness " << topFrameThick <<" positioned at";
  for (int i = 0; i < (int)(topFrameZ.size()); i++)
    LogDebug("TECGeom") << "\t[" << i << "] = " << topFrameZ[i];

  sideFrameMat   = sArgs["SideFrameMaterial"];
  sideFrameThick = nArgs["SideFrameThick"];
  sideFrameZ     = vArgs["SideFrameZ"];
  LogDebug("TECGeom") << "DDTECModuleAlgo debug : Side Frame Material " 
		      << sideFrameMat << " Thickness " << sideFrameThick 
		      << " positioned at";
  for (int i = 0; i < (int)(sideFrameZ.size()); i++)
    LogDebug("TECGeom") << "\t[" << i << "] = " << sideFrameZ[i];

  waferMat       = sArgs["WaferMaterial"];
  sideWidth      = nArgs["SideWidth"];
  waferRot       =vsArgs["WaferRotation"];
  LogDebug("TECGeom") << "DDTECModuleAlgo debug: Wafer Material " 
		      << waferMat << " Side Width " << sideWidth 
		      << " positioned with rotation"	<< " matrix:";
  for (int i=0; i<(int)(waferRot.size()); i++)
    LogDebug("TECGeom") << "\t[" << i << "] = " << waferRot[i];

  activeMat      = sArgs["ActiveMaterial"];
  activeHeight   = nArgs["ActiveHeight"];
  activeThick    = vArgs["ActiveThick"];
  activeRot      = sArgs["ActiveRotation"];
  activeZ        = vArgs["ActiveZ"];
  LogDebug("TECGeom") << "DDTECModuleAlgo debug: Active Material " 
		      << activeMat << " Height " << activeHeight 
		      << " rotated by " << activeRot << " Thickness/Z";
  for (int i=0; i<(int)(activeThick.size()); i++)
    LogDebug("TECGeom") << "\t[" << i << "] = " << activeThick[i] << "/" 
			<< activeZ[i];

  hybridMat      = sArgs["HybridMaterial"];
  hybridHeight   = nArgs["HybridHeight"];
  hybridWidth    = nArgs["HybridWidth"];
  hybridThick    = nArgs["HybridThick"];
  hybridZ        = vArgs["HybridZ"];
  LogDebug("TECGeom") << "DDTECModuleAlgo debug: Hybrid Material " 
		      << hybridMat << " Height " << hybridHeight 
		      << " Width " << hybridWidth << " Thickness " 
		      << hybridThick << " Z";
  for (int i=0; i<(int)(hybridZ.size()); i++)
    LogDebug("TECGeom") << "\t[" << i << "] = " << hybridZ[i];

  pitchMat       = sArgs["PitchMaterial"];
  pitchHeight    = nArgs["PitchHeight"];
  pitchThick     = nArgs["PitchThick"];
  pitchZ         = vArgs["PitchZ"];
  pitchRot       = sArgs["PitchRotation"];
  LogDebug("TECGeom") << "DDTECModuleAlgo debug: Pitch Adapter Material " 
		      << pitchMat << " Height " << pitchHeight 
		      << " Thickness " << pitchThick << " position with "
		      << " rotation " << pitchRot << " at Z";
  for (int i=0; i<(int)(pitchZ.size()); i++)
    LogDebug("TECGeom") << "\t[" << i << "] = " << pitchZ[i];

  bridgeMat      = sArgs["BridgeMaterial"];
  bridgeWidth    = nArgs["BridgeWidth"];
  bridgeThick    = nArgs["BridgeThick"];
  bridgeHeight   = nArgs["BridgeHeight"];
  bridgeSep      = nArgs["BridgeSeparation"];
  LogDebug("TECGeom") << "DDTECModuleAlgo debug: Bridge Material " 
		      << bridgeMat << " Width " << bridgeWidth 
		      << " Thickness " << bridgeThick << " Height " 
		      << bridgeHeight << " Separation "<< bridgeSep;
}

void DDTECModuleAlgo::execute() {
  
  LogDebug("TECGeom") << "==>> Constructing DDTECModuleAlgo...";

  DDName  parentName = parent().name(); 
  std::string idName = DDSplit(parentName).first;
  DDName matname(DDSplit(genMat).first, DDSplit(genMat).second);
  DDMaterial matter(matname);
  double dzdif = fullHeight + topFrameHeight;
  double dxbot, dxtop, topfr;
  if (dlHybrid > dlTop) {
    dxbot = 0.5*dlBottom + frameWidth - frameOver;
    dxtop = 0.5*dlHybrid + frameWidth - frameOver;
    topfr = 0.5*dlBottom * sin(detTilt);
  } else {
    dxbot = 0.5*dlHybrid + frameWidth - frameOver;
    dxtop = 0.5*dlTop    + frameWidth - frameOver;
    topfr = 0.5*dlTop    * sin(detTilt);
  }
  double dxdif = dxtop - dxbot;
  double bl1   = dxbot - dxdif * topfr / (2.0 * dzdif);
  double bl2   = dxtop + dxdif * topfr / (2.0 * dzdif);
  double h1    = 0.5 * frameThick;
  double dz    = 0.5 * dzdif + topfr;
  
  DDSolid solid = DDSolidFactory::trap(DDName(idName,idNameSpace), dz, 0, 0, 
				       h1, bl1, bl1, 0, h1, bl2, bl2, 0);
  LogDebug("TECGeom") << "DDTECModuleAlgo test:\t" << solid.name() 
		      << " Trap made of " << matname << " of dimensions " 
		      << dz << ", 0, 0, " << h1 << ", " << bl1 << ", " 
		      << bl1 << ", 0, " << h1 << ", " << bl2 << ", " 
		      << bl2 << ", 0";
  DDLogicalPart module(solid.ddname(), matter, solid);

  //Top of the frame
  std::string name = idName + "TopFrame";
  matname = DDName(DDSplit(topFrameMat).first, DDSplit(topFrameMat).second);
  matter  = DDMaterial(matname);
  if (dlHybrid > dlTop) {
    bl1 = dxtop - topFrameHeight * dxdif / dzdif;
    bl2 = dxtop;
  } else {
    bl1 = dxbot;
    bl2 = dxbot + topFrameHeight * dxdif / dzdif;
  }
  h1 = 0.5 * topFrameThick;
  dz = 0.5 * topFrameHeight;
  solid = DDSolidFactory::trap(DDName(name,idNameSpace), dz, 0, 0, h1, bl1, 
			       bl1, 0, h1, bl2, bl2, 0);
  LogDebug("TECGeom") << "DDTECModuleAlgo test:\t" << solid.name() 
		      << " Trap made of " << matname << " of dimensions " 
		      << dz << ", 0, 0, " << h1 << ", " << bl1 << ", "  
		      << bl1 << ", 0, " << h1 << ", " << bl2 << ", " << bl2
		      << ", 0";
  DDLogicalPart topFrame(solid.ddname(), matter, solid);

  //Frame Sides
  name    = idName + "SideFrame";
  matname = DDName(DDSplit(sideFrameMat).first, DDSplit(sideFrameMat).second);
  matter  = DDMaterial(matname);
  if (dlHybrid > dlTop) {
    bl2 = bl1;
    bl1 = dxbot;
  } else {
    bl1 = bl2;
    bl2 = dxtop;
  }
  h1 = 0.5 * sideFrameThick;
  dz = 0.5 * fullHeight;
  solid = DDSolidFactory::trap(DDName(name,idNameSpace), dz, 0, 0, h1, bl1, 
			       bl1, 0, h1, bl2, bl2, 0);
  LogDebug("TECGeom") << "DDTECModuleAlgo test:\t" << solid.name() 
		      << " Trap made of " << matname << " of dimensions "
		      << dz << ", 0, 0, " << h1 << ", " << bl1 << ", "
		      << bl1 << ", 0, " << h1 << ", " << bl2 << ", " << bl2
		      << ", 0";
  DDLogicalPart sideFrame(solid.ddname(), matter, solid);

  name    = idName + "Frame";
  matname = DDName(DDSplit(genMat).first, DDSplit(genMat).second);
  matter  = DDMaterial(matname);
  bl1    -= frameWidth;
  bl2    -= frameWidth;
  solid = DDSolidFactory::trap(DDName(name,idNameSpace), dz, 0, 0, h1, bl1, 
			       bl1, 0, h1, bl2, bl2, 0);
  LogDebug("TECGeom") << "DDTECModuleAlgo test:\t" << solid.name() 
		      << " Trap made of " << matname << " of dimensions " 
		      << dz << ", 0, 0, " << h1 << ", " << bl1 << ", " 
		      << bl1 << ", 0, " << h1 << ", " << bl2 << ", " << bl2
		      << ", 0";
  DDLogicalPart frame(solid.ddname(), matter, solid);
  DDpos (frame, sideFrame, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  LogDebug("TECGeom") << "DDTECModuleAlgo test: " << frame.name() 
		      << " number 1 positioned in " << sideFrame.name() 
		      << " at (0,0,0) with no rotation";

  name    = idName + "Hybrid";
  matname = DDName(DDSplit(hybridMat).first, DDSplit(hybridMat).second);
  matter  = DDMaterial(matname);
  double dx = 0.5 * hybridWidth;
  double dy = 0.5 * hybridThick;
  dz        = 0.5 * hybridHeight;
  solid = DDSolidFactory::box(DDName(name, idNameSpace), dx, dy, dz);
  LogDebug("TECGeom") << "DDTECModuleAlgo test:\t" << solid.name() 
		      << " Box made of " << matname << " of dimensions "
		      << dx << ", " << dy << ", " << dz;
  DDLogicalPart hybrid(solid.ddname(), matter, solid);

  // Loop over detectors to be placed
  for (int k = 0; k < (int)(waferRot.size()); k++) {
    std::string tag("Rphi");
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
    LogDebug("TECGeom") << "DDTECModuleAlgo test:\t" << solid.name()
			<< " Trap made of " << matname << " of dimensions "
			<< dz << ", 0, 0, " << h1 << ", " << bl1 << ", "
			<< bl1 << ", 0, " << h1 << ", " << bl2 << ", "
			<< bl2 << ", 0";
    DDLogicalPart wafer(solid.ddname(), matter, solid);
    double zpos;
    double ypos = activeZ[k];
    if (dlHybrid > dlTop) {
      zpos =-0.5 * topFrameHeight;
    } else {
      zpos = 0.5 * topFrameHeight;
    }
    DDTranslation tran(0, ypos, zpos);
    std::string rotstr = DDSplit(waferRot[k]).first;
    std::string rotns;
    DDRotation rot;
    if (rotstr != "NULL") {
      rotns = DDSplit(waferRot[k]).second;
      rot   = DDRotation(DDName(rotstr, rotns));
    }
    DDpos (wafer, module, k+1, tran, rot);
    LogDebug("TECGeom") << "DDTECModuleAlgo test: " << wafer.name() 
			<< " number " << k+1 << " positioned in " 
			<< module.name() << " at " << tran << " with " 
			<< rot;

    // Active
    name    = idName + tag + "Active";
    matname = DDName(DDSplit(activeMat).first, DDSplit(activeMat).second);
    matter  = DDMaterial(matname);
    bl1    -= sideWidth;
    bl2    -= sideWidth;
    dz      = 0.5 * activeThick[k];
    h1      = 0.5 * activeHeight;
    double bl1p = bl1;
    double bl2p = bl2;
    if (dlHybrid > dlTop) {
      bl1p = bl2;
      bl2p = bl1;
    }
    solid = DDSolidFactory::trap(DDName(name,idNameSpace), dz, 0, 0, h1, bl1p, 
				 bl2p, 0, h1, bl1p, bl2p, 0);
    LogDebug("TECGeom") << "DDTECModuleAlgo test:\t" << solid.name() 
			<< " Trap made of " << matname << " of dimensions "
			<< dz << ", 0, 0, " << h1 << ", " << bl1p << ", "
			<< bl2p << ", 0, " << h1 << ", " << bl1p << ", "
			<< bl2p << ", 0";
    DDLogicalPart active(solid.ddname(), matter, solid);
    rotstr = DDSplit(activeRot).first;
    rot    = DDRotation();
    if (rotstr != "NULL") {
      rotns = DDSplit(activeRot).second;
      rot   = DDRotation(DDName(rotstr, rotns));
    }
    DDpos (active, wafer, 1, DDTranslation(0.0, 0.0, 0.0), rot);
    LogDebug("TECGeom") << "DDTECModuleAlgo test: " << active.name()
			<< " number 1 positioned in " << wafer.name()
			<< " at (0, 0, 0) with " << rot;

    //Pitch Adapter
    name    = idName + tag + "PA";
    matname = DDName(DDSplit(pitchMat).first, DDSplit(pitchMat).second);
    matter  = DDMaterial(matname);
    if (dlHybrid > dlTop) {
      dz   = 0.5 * dlTop;
      zpos = 0.5 * (dzdif - pitchHeight) - hybridHeight;
    } else {
      dz   = 0.5 * dlBottom;
      zpos =-0.5 * (dzdif - pitchHeight) + hybridHeight;
    }
    ypos = pitchZ[k];
    double xpos = 0;
    if (k == 0) {
      dx      = dz;
      dy      = 0.5 * pitchThick;
      dz      = 0.5 * pitchHeight;
      solid   = DDSolidFactory::box(DDName(name, idNameSpace), dx, dy, dz);
      LogDebug("TECGeom") << "DDTECModuleAlgo test:\t" << solid.name() 
			  << " Box made of " << matname <<" of dimensions "
			  << dx << ", " << dy << ", " << dz;
      rot     = DDRotation();
    } else {
      h1      = 0.5 * pitchThick;
      bl1     = 0.5 * pitchHeight + 0.5 * dz * sin(detTilt);
      bl2     = 0.5 * pitchHeight - 0.5 * dz * sin(detTilt);
      double thet = atan((bl1-bl2)/(2.*dz));
      solid   = DDSolidFactory::trap(DDName(name,idNameSpace), dz, thet, 0, h1,
				     bl1, bl1, 0, h1, bl2, bl2, 0);
      LogDebug("TECGeom") << "DDTECModuleAlgo test:\t" << solid.name()
			  << " Trap made of " << matname 
			  << " of dimensions " << dz << ", " << thet/deg
			  << ", 0, " << h1 << ", " << bl1 << ", " << bl1
			  << ", 0, " << h1 << ", " << bl2 << ", " << bl2
			  << ", 0";
      xpos    = 0.5 * fullHeight * sin(detTilt);
      rotstr  = DDSplit(pitchRot).first;
      rotns   = DDSplit(pitchRot).second;
      rot     = DDRotation(DDName(rotstr, rotns));
    }
    DDLogicalPart pa(solid.ddname(), matter, solid);
    tran = DDTranslation(xpos,ypos,zpos);
    DDpos (pa, module, k+1, tran, rot);
    LogDebug("TECGeom") << "DDTECModuleAlgo test: " << pa.name() 
			<< " number " << k+1 << " positioned in "
			<< module.name() << " at " << tran << " with "
			<< rot;

    // Position the hybrid now
    ypos = hybridZ[k];
    if (dlHybrid > dlTop) {
      zpos = 0.5 * (dzdif - hybridHeight);
    } else {
      zpos =-0.5 * (dzdif - hybridHeight);
    }
    tran = DDTranslation(0,ypos,zpos);
    rot  = DDRotation();
    DDpos (hybrid, module, k+1, tran, rot);
    LogDebug("TECGeom") << "DDTECModuleAlgo test: " << hybrid.name()
			<< " number "  << k+1 << " positioned in "
			<< module.name() << " at " << tran << " with " << rot;

    // Position the frame
    ypos = topFrameZ[k];
    if (dlHybrid > dlTop) {
      zpos = 0.5 * (dzdif - topFrameHeight);
    } else {
      zpos =-0.5 * (dzdif - topFrameHeight);
    }
    tran = DDTranslation(0,ypos,zpos);
    rot  = DDRotation();
    DDpos (topFrame, module, k+1, tran, rot);
    LogDebug("TECGeom") << "DDTECModuleAlgo test: " << topFrame.name()
			<< " number " << k+1 << " positioned in "
			<< module.name() << " at " << tran << " with " << rot;

    ypos = sideFrameZ[k];
    if (dlHybrid > dlTop) {
      zpos =-0.5 * topFrameHeight;
    } else {
      zpos = 0.5 * topFrameHeight;
    }
    tran = DDTranslation(0,ypos,zpos);
    rot  = DDRotation();
    DDpos (sideFrame, module, k+1, tran, rot);
    LogDebug("TECGeom") << "DDTECModuleAlgo test: " << sideFrame.name()
			<< " number " << k+1 << " positioned in " 
			<< module.name() << " at " << tran << " with " << rot;
  }

  //Bridge 
  name    = idName + "Bridge";
  matname = DDName(DDSplit(bridgeMat).first, DDSplit(bridgeMat).second);
  matter  = DDMaterial(matname);
  bl2     = 0.5*bridgeSep + bridgeWidth;
  bl1     = bl2 - bridgeHeight * dxdif / dzdif;
  h1      = 0.5 * bridgeThick;
  dz      = 0.5 * bridgeHeight;
  solid = DDSolidFactory::trap(DDName(name,idNameSpace), dz, 0, 0, h1, bl1, 
			       bl1, 0, h1, bl2, bl2, 0);
  LogDebug("TECGeom") << "DDTECModuleAlgo test:\t" << solid.name()
		      << " Trap made of " << matname << " of dimensions "
		      << dz << ", 0, 0, " << h1 << ", " << bl1 << ", "
		      << bl1 << ", 0, " << h1 << ", " << bl2 << ", " << bl2
		      << ", 0";
  DDLogicalPart bridge(solid.ddname(), matter, solid);

  name    = idName + "BridgeGap";
  matname = DDName(DDSplit(genMat).first, DDSplit(genMat).second);
  matter  = DDMaterial(matname);
  bl1     = 0.5*bridgeSep;
  solid = DDSolidFactory::box(DDName(name,idNameSpace), bl1, h1, dz);
  LogDebug("TECGeom") << "DDTECModuleAlgo test:\t" << solid.name() 
		      << " Box made of " << matname << " of dimensions "
		      << bl1 << ", " << h1 << ", " << dz;
  DDLogicalPart bridgeGap(solid.ddname(), matter, solid);
  DDpos (bridgeGap, bridge, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());
  LogDebug("TECGeom") << "DDTECModuleAlgo test: " << bridgeGap.name() 
		      << " number 1 positioned in " << bridge.name()
		      << " at (0,0,0) with no rotation";

  LogDebug("TECGeom") << "<<== End of DDTECModuleAlgo construction ...";
}
