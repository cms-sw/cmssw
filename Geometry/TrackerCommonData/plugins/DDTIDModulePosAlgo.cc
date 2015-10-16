///////////////////////////////////////////////////////////////////////////////
// File: DDTIDModulePosAlgo.cc
// Description: Position various components inside a TID Module
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTIDModulePosAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDTIDModulePosAlgo::DDTIDModulePosAlgo() {
  LogDebug("TIDGeom") << "DDTIDModulePosAlgo info: Creating an instance";
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
		      << detTilt/CLHEP::deg << " Height " << fullHeight 
		      << " dl(Top) " << dlTop << " dl(Bottom) " << dlBottom
		      << " dl(Hybrid) " << dlHybrid;

  boxFrameName      = sArgs["BoxFrameName"];
  boxFrameHeight    = nArgs["BoxFrameHeight"];
  boxFrameWidth     = nArgs["BoxFrameWidth"];
  boxFrameZ         = vArgs["BoxFrameZ"];
  bottomFrameHeight = nArgs["BottomFrameHeight"];
  bottomFrameOver   = nArgs["BottomFrameOver"];
  topFrameHeight    = nArgs["TopFrameHeight"];
  topFrameOver      = nArgs["TopFrameOver"];
  LogDebug("TIDGeom") << "DDTIDModulePosAlgo debug: " << boxFrameName 
		      << " positioned at Z";
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "\tboxFrameZ[" << i << "] = " << boxFrameZ[i];
  LogDebug("TIDGeom") << "\t Extra Height at Bottom " << bottomFrameHeight
		      << " Overlap " <<bottomFrameOver;

  sideFrameName     = vsArgs["SideFrameName"];
  sideFrameZ        = vArgs["SideFrameZ"];
  sideFrameRot      =vsArgs["SideFrameRotation"];
  sideFrameWidth    = nArgs["SideFrameWidth"];
  sideFrameOver     = nArgs["SideFrameOver"];
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "\tsideFrame[" << i << "] = " << sideFrameName[i]
			<< " positioned at Z "<< sideFrameZ[i]
			<< " with rotation " << sideFrameRot[i];

  kaptonName     = vsArgs["KaptonName"];
  kaptonZ        = vArgs["KaptonZ"];
  kaptonRot      =vsArgs["KaptonRotation"];
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "\tkapton[" << i << "] = " << kaptonName[i]
			<< " positioned at Z "<< kaptonZ[i]
			<< " with rotation " << kaptonRot[i];

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

  coolName         = sArgs["CoolInsertName"];
  coolHeight       = nArgs["CoolInsertHeight"];
  coolZ            = nArgs["CoolInsertZ"];
  coolWidth        = nArgs["CoolInsertWidth"];
  coolRadShift     = vArgs["CoolInsertShift"];

  std::string comp  = sArgs["DoSpacers"];
  if (comp == "No" || comp == "NO" || comp == "no") doSpacers = false;
  else                                              doSpacers = true;

  botSpacersName         = sArgs["BottomSpacersName"];
  botSpacersHeight       = nArgs["BottomSpacersHeight"];
  botSpacersZ            = nArgs["BottomSpacersZ"];
  sidSpacersName         = sArgs["SideSpacersName"];
  sidSpacersHeight       = nArgs["SideSpacersHeight"];
  sidSpacersZ            = nArgs["SideSpacersZ"];
  sidSpacersWidth        = nArgs["SideSpacersWidth"];
  sidSpacersRadShift     = nArgs["SideSpacersShift"];

}

void DDTIDModulePosAlgo::execute(DDCompactView& cpv) {
  
  LogDebug("TIDGeom") << "==>> Constructing DDTIDModulePosAlgo...";

  DDName parentName  = parent().name(); 
  DDName name;

  double botfr;                                       // width of side frame at the the bottom of the modules 
  double topfr;                                       // width of side frame at the the top of the modules 
  double kaptonHeight;
  if (dlHybrid > dlTop) {
    // ring 1, ring 2
    topfr = topFrameHeight - pitchHeight - topFrameOver;      
    botfr = bottomFrameHeight - bottomFrameOver; 
    kaptonHeight = fullHeight + botfr;
  } else {
    // ring 3
    topfr = topFrameHeight - topFrameOver;      
    botfr = bottomFrameHeight - bottomFrameOver - pitchHeight; 
    kaptonHeight = fullHeight + topfr;
  }

  double sideFrameHeight = fullHeight + pitchHeight + botfr + topfr; 
  double zCenter     = 0.5 * (sideFrameHeight+boxFrameHeight); 


  // (Re) Compute the envelope for positioning Cool Inserts and Side Spacers (Alumina).
  double  sidfr = sideFrameWidth - sideFrameOver;      // width of side frame on the sides of module 
  double  dxbot = 0.5*dlBottom + sidfr;
  double  dxtop = 0.5*dlTop + sidfr;
  double  dxtopenv, dxbotenv;           // top/bot width of the module envelope trap

  double tanWafer=(dxtop-dxbot)/fullHeight; // 
  double thetaWafer = atan(tanWafer);       // 1/2 of the wafer wedge angle

  if (dlHybrid > dlTop) {
    // ring 1, ring 2
    dxtopenv = dxbot + (dxtop-dxbot)*(fullHeight+pitchHeight+topfr+hybridHeight)/fullHeight;
    dxbotenv = dxtop - (dxtop-dxbot)*(fullHeight+botfr)/fullHeight;
  } else {
    // ring 3
    dxtopenv = dxbot + (dxtop-dxbot)*(fullHeight+topfr)/fullHeight;
    dxbotenv = dxbot;
  }

  double tanEnv=(dxtopenv-dxbotenv)/(sideFrameHeight+boxFrameHeight); // 1/2 of the envelope wedge angle

  double xpos=0; double ypos=0; double zpos=0;

  // Cool Inserts
  name = DDName(DDSplit(coolName).first, DDSplit(coolName).second);
  ypos = coolZ;

  double zCool;
  int copy=0;
  DDRotation rot  = DDRotation(); // should be different for different elements

  for (int j1=0; j1<2; j1++){  // j1: 0 inserts below the hybrid
                               //     1 inserts below the wafer
    if (dlHybrid > dlTop) {
      zCool = sideFrameHeight+boxFrameHeight-coolRadShift[j1];  
      if ( j1==0 ) zCool -= 0.5*coolHeight; 
    } else {
      zCool = coolRadShift[j1];  
      if ( j1==0 ) zCool += 0.5*coolHeight; 
    }

    if ( j1==0 ) {
      xpos = -0.5*(boxFrameWidth-coolWidth);
    } else {   
      xpos = -(dxbotenv+(zCool-0.5*coolHeight)*tanEnv-0.5*coolWidth);      
    }
		   
    zpos = zCool-zCenter;
    for ( int j2=0; j2<2; j2++) {
      copy++;
     cpv.position(name, parentName, copy,  DDTranslation(xpos,ypos,zpos), rot);
      LogDebug("TIDGeom") << "DDTIDModulePosAlgo test: " << name <<" number "
			  << copy << " positioned in " << parentName << " at "
			  << DDTranslation(xpos,ypos,zpos) << " with " << rot;
      xpos = -xpos;
    }
  }


  if ( doSpacers ) {
  // Bottom Spacers (Alumina)
    name = DDName(DDSplit(botSpacersName).first, DDSplit(botSpacersName).second);
    ypos = botSpacersZ;

    double zBotSpacers;
    if (dlHybrid > dlTop) {
      zBotSpacers = sideFrameHeight+boxFrameHeight-0.5*botSpacersHeight;
    } else {
      zBotSpacers = 0.5*botSpacersHeight;
    }
    zpos = zBotSpacers - zCenter; 
    rot = DDRotation();
   cpv.position(name, parentName, 1,  DDTranslation(0.0,ypos,zpos), rot );
    LogDebug("TIDGeom") << "DDTIDModulePosAlgo test: " << name <<" number "
			<< 1 << " positioned in " << parentName << " at "
			<< DDTranslation(0.0,ypos,zpos) << " with no rotation";       	

    
    // Side Spacers (Alumina)
    name = DDName(DDSplit(sidSpacersName).first, 
		  DDSplit(sidSpacersName).second);
    ypos = sidSpacersZ;

    double zSideSpacers;
    if (dlHybrid > dlTop) {
      zSideSpacers = sideFrameHeight+boxFrameHeight-sidSpacersRadShift;
    } else {
      zSideSpacers = sidSpacersRadShift;
    }
    zpos = zSideSpacers - zCenter; 
    
    copy=0;
    xpos = dxbotenv+(zSideSpacers-0.5*sidSpacersHeight)*tanEnv-0.5*sidSpacersWidth+sideFrameOver;      

    double phix, phiy, phiz;
    phix=0.*CLHEP::deg; phiy=90.*CLHEP::deg; phiz=0.*CLHEP::deg;

    double thetay, thetax;
    thetay=90.*CLHEP::deg;
    double thetaz = thetaWafer;

    for (int j1=0; j1<2; j1++){
      copy++;
 
      // tilt Side Spacers (parallel to Side Frame)
      thetax = 90.*CLHEP::deg+thetaz;
      double thetadeg = thetax/CLHEP::deg;
      if (thetadeg != 0) {
	std::string arotstr = DDSplit(sidSpacersName).first+dbl_to_string(thetadeg*10.);
	rot = DDrot(DDName(arotstr,  DDSplit(sidSpacersName).second), thetax, 
		    phix, thetay, phiy, thetaz, phiz);
      }

     cpv.position(name, parentName, copy,  DDTranslation(xpos,ypos,zpos), rot);
      LogDebug("TIDGeom") << "DDTIDModulePosAlgo test: " << name <<" number "
			  << copy << " positioned in " << parentName << " at "
			  << DDTranslation(xpos,ypos,zpos) << " with " << rot;
      xpos = -xpos;
      thetaz = -thetaz;
    }
  }

  // Loop over detectors to be placed
  for (int k = 0; k < detectorN; k++) {
    // Wafer
    name = DDName(DDSplit(waferName[k]).first, DDSplit(waferName[k]).second);
    xpos=0; 
    ypos = waferZ[k];
    double zWafer;
    if (dlHybrid > dlTop) {
      zWafer = botfr+0.5*fullHeight;
    } else {
      zWafer = boxFrameHeight+botfr+pitchHeight+0.5*fullHeight;
    }
    zpos = zWafer - zCenter;
    DDTranslation tran(xpos, ypos, zpos);
    std::string rotstr = DDSplit(waferRot[k]).first;
    std::string rotns;
    if (rotstr != "NULL") {
      rotns = DDSplit(waferRot[k]).second;
      rot   = DDRotation(DDName(rotstr, rotns));
    }
   cpv.position(name, parentName, k+1, tran, rot);
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
    double zPitch;
    if (dlHybrid > dlTop) {
      zPitch = botfr+fullHeight+0.5*pitchHeight;
    } else {
      zPitch = boxFrameHeight+botfr+0.5*pitchHeight;
    }
    zpos = zPitch - zCenter;
    rotstr  = DDSplit(pitchRot[k]).first;
    if (rotstr != "NULL") {
      rotns = DDSplit(pitchRot[k]).second;
      rot   = DDRotation(DDName(rotstr, rotns));
    } else {
      rot     = DDRotation();
    }
    tran = DDTranslation(xpos,ypos,zpos);
   cpv.position(name, parentName, k+1, tran, rot);
    LogDebug("TIDGeom") << "DDTIDModulePosAlgo test: " << name <<" number "
			<< k+1 << " positioned in " << parentName << " at "
			<< tran << " with " << rot;

    // Hybrid 
    name = DDName(DDSplit(hybridName).first, DDSplit(hybridName).second);
    ypos = hybridZ[k];
    double zHybrid;
    if (dlHybrid > dlTop) {
      zHybrid = botfr+fullHeight+pitchHeight+0.5*hybridHeight;
    } else {
      zHybrid = 0.5*hybridHeight;
    }
    zpos = zHybrid - zCenter;
    tran = DDTranslation(0,ypos,zpos);
    rot  = DDRotation();
   cpv.position(name, parentName, k+1, tran, rot);
    LogDebug("TIDGeom") << "DDTIDModulePosAlgo test: " << name <<" number "
			<< k+1 << " positioned in " << parentName << " at "
			<< tran << " with " << rot;


    // Box frame
    name = DDName(DDSplit(boxFrameName).first, DDSplit(boxFrameName).second);
    ypos = boxFrameZ[k];
    double zBoxFrame;
    if (dlHybrid > dlTop) {
      zBoxFrame = sideFrameHeight+0.5*boxFrameHeight;
    } else {
      zBoxFrame = 0.5*boxFrameHeight;
    }
    zpos = zBoxFrame - zCenter;
    tran = DDTranslation(0,ypos,zpos);
    rot  = DDRotation();
   cpv.position(name, parentName, k+1, tran, rot);
    LogDebug("TIDGeom") << "DDTIDModulePosAlgo test: " << name <<" number "
			<< k+1 << " positioned in " << parentName << " at "
			<< tran << " with " << rot;

    // Side frame
    name = DDName(DDSplit(sideFrameName[k]).first, 
		  DDSplit(sideFrameName[k]).second);
    ypos = sideFrameZ[k];
    double zSideFrame;
    if (dlHybrid > dlTop) {
      zSideFrame = 0.5*sideFrameHeight;
    } else {
      zSideFrame = boxFrameHeight+0.5*sideFrameHeight;
    }
    zpos = zSideFrame-zCenter;
    rotstr  = DDSplit(sideFrameRot[k]).first;
    if (rotstr != "NULL") {
      rotns = DDSplit(sideFrameRot[k]).second;
      rot   = DDRotation(DDName(rotstr, rotns));
    } else {
      rot     = DDRotation();
    }  
    tran = DDTranslation(0,ypos,zpos);
   cpv.position(name, parentName, k+1, tran, rot);
    LogDebug("TIDGeom") << "DDTIDModulePosAlgo test: " << name <<" number "
			<< k+1 << " positioned in " << parentName << " at "
			<< tran << " with " << rot;


    // Kapton circuit
    name = DDName(DDSplit(kaptonName[k]).first, DDSplit(kaptonName[k]).second);
    ypos = kaptonZ[k];
    double zKapton;
    double kaptonExtraHeight=0;
    if (dlHybrid > dlTop) {
      if ( k == 1 ) kaptonExtraHeight = dlTop*sin(detTilt)-fullHeight*(1-cos(detTilt));
      kaptonExtraHeight = 0.5*fabs(kaptonExtraHeight);
      zKapton = 0.5*(kaptonHeight+kaptonExtraHeight);
    } else {
      if ( k == 1 ) kaptonExtraHeight = dlBottom*sin(detTilt)-fullHeight*(1-cos(detTilt));
      kaptonExtraHeight = 0.5*fabs(kaptonExtraHeight);
      zKapton = boxFrameHeight+sideFrameHeight-0.5*(kaptonHeight+kaptonExtraHeight);
    }
    zpos = zKapton-zCenter;
    rotstr  = DDSplit(kaptonRot[k]).first;
    if (rotstr != "NULL") {
      rotns = DDSplit(kaptonRot[k]).second;
      rot   = DDRotation(DDName(rotstr, rotns));
    } else {
      rot     = DDRotation();
    }  
    tran = DDTranslation(0,ypos,zpos);
   cpv.position(name, parentName, k+1, tran, rot);
    LogDebug("TIDGeom") << "DDTIDModulePosAlgo test: " << name <<" number "
			<< k+1 << " positioned in " << parentName << " at "
			<< tran << " with " << rot;
  }

  LogDebug("TIDGeom") << "<<== End of DDTIDModulePosAlgo positioning ...";
}


