///////////////////////////////////////////////////////////////////////////////
// File: DDTIDModulePosAlgo.cc
// Description: Position various components inside a TID Module
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/interface/DDTIDModulePosAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTIDModulePosAlgo::DDTIDModulePosAlgo() {
  edm::LogInfo("TIDGeom") << "DDTIDModulePosAlgo info: Creating an "
			  << "instance";
}

DDTIDModulePosAlgo::~DDTIDModulePosAlgo() {}

void DDTIDModulePosAlgo::initialize(const DDNumericArguments & nArgs,
				    const DDVectorArguments & vArgs,
				    const DDMapArguments & ,
				    const DDStringArguments & sArgs,
				    const DDStringVectorArguments & vsArgs) {

  int i;
  DDName parentName = parent().name(); 
  detectorN         = (int)(nArgs["DetectorNumber"]);

  LogDebug("TIDGeom") << "DDTIDModulePosAlgo debug: Parent " << parentName
		      << " Detector Planes " << detectorN;

  detTilt           = nArgs["DetTilt"];
  fullHeight        = nArgs["FullHeight"];
  dlTop             = nArgs["DlTop"];
  dlBottom          = nArgs["DlBottom"];
  dlHybrid          = nArgs["DlHybrid"];

  LogDebug("TIDGeom") << "DDTIDModulePosAlgo debug: Detector Tilt " 
		      << detTilt/deg << " Height " << fullHeight 
		      << " dl(Top) " << dlTop << " dl(Bottom) " << dlBottom
		      << " dl(Hybrid) " << dlHybrid;

  topFrameName      = sArgs["TopFrameName"];
  topFrameHeight    = nArgs["TopFrameHeight"];
  topFrameZ         = vArgs["TopFrameZ"];
  bottomFrameHeight = nArgs["BottomFrameHeight"];
  bottomFrameOver   = nArgs["BottomFrameOver"];
  LogDebug("TIDGeom") << "DDTIDModulePosAlgo debug: " << topFrameName 
		      << " Height " << topFrameHeight <<" positioned at Z";
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "\ttopFrameZ[" << i << "] = " << topFrameZ[i];
  LogDebug("TIDGeom") << "\t Extra Height at Bottom " << bottomFrameHeight
		      << " Overlap " <<bottomFrameOver;

  sideFrameName     = sArgs["SideFrameName"];
  sideFrameZ        = vArgs["SideFrameZ"];
  LogDebug("TIDGeom") << "DDTIDModulePosAlgo debug : " << sideFrameName
		      << " positioned at Z";
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "\tsideFrameZ[" << i << "] = " << sideFrameZ[i];

  waferName         =vsArgs["WaferName"];
  waferZ            = vArgs["WaferZ"];
  waferRot          =vsArgs["WaferRotation"];
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "DDTIDModulePosAlgo debug: " << waferName[i]
			<< " positioned at Z " << waferZ[i] 
			<< " with rotation " << waferRot[i];

  hybridName        = sArgs["HybridName"];
  hybridHeight      = nArgs["HybridHeight"];
  hybridZ           = vArgs["HybridZ"];
  LogDebug("TIDGeom") << "DDTIDModulePosAlgo debug: " << hybridName 
		      << " Height " << hybridHeight << " Z";
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "\thybridZ[" << i <<"] = " << hybridZ[i];

  pitchName         =vsArgs["PitchName"];
  pitchHeight       = nArgs["PitchHeight"];
  pitchZ            = vArgs["PitchZ"];
  pitchRot          =vsArgs["PitchRotation"];
  LogDebug("TIDGeom") << "DDTIDModulePosAlgo debug: Pitch Adapter Height " 
		      << pitchHeight;
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "DDTIDModulePosAlgo debug: " << pitchName[i]
			<< " position at Z " << pitchZ[i] 
			<< " with rotation " << pitchRot[i];
}

void DDTIDModulePosAlgo::execute() {
  
  LogDebug("TIDGeom") << "==>> Constructing DDTIDModulePosAlgo...";

  DDName parentName  = parent().name(); 
  double dzdif       = fullHeight + topFrameHeight;
  double topfr       = bottomFrameHeight - bottomFrameOver;

  // Loop over detectors to be placed
  for (int k = 0; k < detectorN; k++) {
    DDName name;

    // Wafer
    name = DDName(DDSplit(waferName[k]).first, DDSplit(waferName[k]).second);
    double zpos, xpos=0;
    double ypos = waferZ[k];
    if (dlHybrid > dlTop) {
      zpos =-0.5*(topFrameHeight-topfr);
    } else {
      zpos = 0.5*(topFrameHeight-topfr);
    }
    DDTranslation tran(xpos, ypos, zpos);
    std::string rotstr = DDSplit(waferRot[k]).first;
    std::string rotns;
    DDRotation rot;
    if (rotstr != "NULL") {
      rotns = DDSplit(waferRot[k]).second;
      rot   = DDRotation(DDName(rotstr, rotns));
    }
    DDpos (name, parentName, k+1, tran, rot);
    LogDebug("TIDGeom") << "DDTIDModulePosAlgo test: " << name <<" number "
			<< k+1 << " positioned in " << parentName << " at "
			<< tran << " with " << rot;

    //Pitch Adapter
    name = DDName(DDSplit(pitchName[k]).first, DDSplit(pitchName[k]).second);
    if (k == 0) {
      xpos = 0;
    } else {
      xpos = 0.5 * fullHeight * sin(detTilt);
    }
    ypos = pitchZ[k];
    if (dlHybrid > dlTop) {
      zpos = 0.5 * (dzdif+topfr-pitchHeight) - hybridHeight;
    } else {
      zpos =-0.5 * (dzdif+topfr-pitchHeight) + hybridHeight;
    }
    rotstr  = DDSplit(pitchRot[k]).first;
    if (rotstr != "NULL") {
      rotns = DDSplit(pitchRot[k]).second;
      rot   = DDRotation(DDName(rotstr, rotns));
    } else {
      rot     = DDRotation();
    }
    tran = DDTranslation(xpos,ypos,zpos);
    DDpos (name, parentName, k+1, tran, rot);
    LogDebug("TIDGeom") << "DDTIDModulePosAlgo test: " << name <<" number "
			<< k+1 << " positioned in " << parentName << " at "
			<< tran << " with " << rot;

    // Position the hybrid now
    name = DDName(DDSplit(hybridName).first, DDSplit(hybridName).second);
    ypos = hybridZ[k];
    if (dlHybrid > dlTop) {
      zpos = 0.5 * (dzdif+topfr-hybridHeight);
    } else {
      zpos =-0.5 * (dzdif+topfr-hybridHeight);
    }
    tran = DDTranslation(0,ypos,zpos);
    rot  = DDRotation();
    DDpos (name, parentName, k+1, tran, rot);
    LogDebug("TIDGeom") << "DDTIDModulePosAlgo test: " << name <<" number "
			<< k+1 << " positioned in " << parentName << " at "
			<< tran << " with " << rot;

    // Position the top frame
    name = DDName(DDSplit(topFrameName).first, DDSplit(topFrameName).second);
    ypos = topFrameZ[k];
    if (dlHybrid > dlTop) {
      zpos = 0.5 * (dzdif+topfr-topFrameHeight);
    } else {
      zpos =-0.5 * (dzdif+topfr-topFrameHeight);
    }
    tran = DDTranslation(0,ypos,zpos);
    rot  = DDRotation();
    DDpos (name, parentName, k+1, tran, rot);
    LogDebug("TIDGeom") << "DDTIDModulePosAlgo test: " << name <<" number "
			<< k+1 << " positioned in " << parentName << " at "
			<< tran << " with " << rot;

    // Position the side frame
    name = DDName(DDSplit(sideFrameName).first, DDSplit(sideFrameName).second);
    ypos = sideFrameZ[k];
    if (dlHybrid > dlTop) {
      zpos =-0.5 * topFrameHeight;
    } else {
      zpos = 0.5 * topFrameHeight;
    }
    tran = DDTranslation(0,ypos,zpos);
    rot  = DDRotation();
    DDpos (name, parentName, k+1, tran, rot);
    LogDebug("TIDGeom") << "DDTIDModulePosAlgo test: " << name <<" number "
			<< k+1 << " positioned in " << parentName << " at "
			<< tran << " with " << rot;
  }

  LogDebug("TIDGeom") << "<<== End of DDTIDModulePosAlgo positioning ...";
}
