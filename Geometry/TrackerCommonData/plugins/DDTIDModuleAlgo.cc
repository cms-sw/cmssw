///////////////////////////////////////////////////////////////////////////////
// File: DDTIDModuleAlgo.cc
// Description: Creation of a TID Module
///////////////////////////////////////////////////////////////////////////////
#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTIDModuleAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


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
  DDName parentName(parent().name()); 

  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug: Parent " << parentName 
		      << " General Material " << genMat 
		      << " Detector Planes " << detectorN;

  moduleThick       = nArgs["ModuleThick"];
  detTilt           = nArgs["DetTilt"];
  fullHeight        = nArgs["FullHeight"];
  dlTop             = nArgs["DlTop"];
  dlBottom          = nArgs["DlBottom"];
  dlHybrid          = nArgs["DlHybrid"];
  std::string comp  = sArgs["DoComponents"];
  if (comp == "No" || comp == "NO" || comp == "no") doComponents = false;
  else                                              doComponents = true;

  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug: ModuleThick " 
		      << moduleThick << " Detector Tilt " << detTilt/CLHEP::deg
		      << " Height " << fullHeight << " dl(Top) " << dlTop
		      << " dl(Bottom) " << dlBottom << " dl(Hybrid) "
		      << dlHybrid << " doComponents " << doComponents;

  boxFrameName      = sArgs["BoxFrameName"];
  boxFrameMat       = sArgs["BoxFrameMaterial"];
  boxFrameThick     = nArgs["BoxFrameThick"];
  boxFrameHeight    = nArgs["BoxFrameHeight"];
  boxFrameWidth     = nArgs["BoxFrameWidth"];
  bottomFrameHeight = nArgs["BottomFrameHeight"];
  bottomFrameOver   = nArgs["BottomFrameOver"];
  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug: " << boxFrameName 
		      << " Material " << boxFrameMat << " Thickness " 
		      << boxFrameThick << " width " << boxFrameWidth 
		      <<  " height " << boxFrameHeight
		      << " Extra Height at Bottom " << bottomFrameHeight 
		      << " Overlap " << bottomFrameOver;

  topFrameHeight    = nArgs["TopFrameHeight"];
  topFrameOver      = nArgs["TopFrameOver"];
  sideFrameName     = vsArgs["SideFrameName"];
  sideFrameMat      = sArgs["SideFrameMaterial"];
  sideFrameWidth    = nArgs["SideFrameWidth"];
  sideFrameThick    = nArgs["SideFrameThick"];
  sideFrameOver     = nArgs["SideFrameOver"];
  holeFrameName     = vsArgs["HoleFrameName"];
  holeFrameRot      = vsArgs["HoleFrameRotation"];
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "DDTIDModuleAlgo debug : " << sideFrameName[i] 
			<< " Material " << sideFrameMat << " Width " 
			<< sideFrameWidth << " Thickness " << sideFrameThick
			<< " Overlap " << sideFrameOver << " Hole  "
			<< holeFrameName[i];

  kaptonName     = vsArgs["KaptonName"];
  kaptonMat      = sArgs["KaptonMaterial"];
  kaptonThick    = nArgs["KaptonThick"];
  kaptonOver     = nArgs["KaptonOver"];
  holeKaptonName     = vsArgs["HoleKaptonName"];
  holeKaptonRot      = vsArgs["HoleKaptonRotation"];
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "DDTIDModuleAlgo debug : " << kaptonName[i] 
			<< " Material " << kaptonMat 
                        << " Thickness " << kaptonThick
			<< " Overlap " << kaptonOver << " Hole  "
			<< holeKaptonName[i];


  waferName         = vsArgs["WaferName"];
  waferMat          = sArgs["WaferMaterial"];
  sideWidthTop      = nArgs["SideWidthTop"];
  sideWidthBottom   = nArgs["SideWidthBottom"];

  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug: Wafer Material " 
		      << waferMat  << " Side Width Top " << sideWidthTop
		      << " Side Width Bottom " << sideWidthBottom;
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "\twaferName[" << i << "] = " << waferName[i];

  activeName        = vsArgs["ActiveName"];
  activeMat         = sArgs["ActiveMaterial"];
  activeHeight      = nArgs["ActiveHeight"];
  waferThick        = vArgs["WaferThick"];
  activeRot         = sArgs["ActiveRotation"];
  backplaneThick    = vArgs["BackPlaneThick"];
  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug: Active Material " 
		      << activeMat << " Height " << activeHeight 
		      << " rotated by " << activeRot;
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << " translated by (0," << -0.5*backplaneThick[i] 
			<< ",0)\tactiveName[" << i << "] = " << activeName[i]
			<< " of thickness " << waferThick[i]-backplaneThick[i];
  
  hybridName        = sArgs["HybridName"];
  hybridMat         = sArgs["HybridMaterial"];
  hybridHeight      = nArgs["HybridHeight"];
  hybridWidth       = nArgs["HybridWidth"];
  hybridThick       = nArgs["HybridThick"];
  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug: " << hybridName 
		      << " Material " << hybridMat << " Height " 
		      << hybridHeight << " Width " << hybridWidth 
		      << " Thickness " << hybridThick;

  pitchName         = vsArgs["PitchName"];
  pitchMat          = sArgs["PitchMaterial"];
  pitchHeight       = nArgs["PitchHeight"];
  pitchThick        = nArgs["PitchThick"];
  pitchStereoTol    = nArgs["PitchStereoTolerance"];

  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug: Pitch Adapter Material "
		      << pitchMat << " Height " << pitchHeight
		      << " Thickness " << pitchThick;
  for (i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") <<  "\tpitchName[" << i << "] = " << pitchName[i];

  coolName         = sArgs["CoolInsertName"];
  coolMat          = sArgs["CoolInsertMaterial"];
  coolHeight       = nArgs["CoolInsertHeight"];
  coolThick        = nArgs["CoolInsertThick"];
  coolWidth        = nArgs["CoolInsertWidth"];
  LogDebug("TIDGeom") << "DDTIDModuleAlgo debug: Cool Element Material "
		      << coolMat << " Height " << coolHeight
		      << " Thickness " << coolThick << " Width " << coolWidth;
}

void DDTIDModuleAlgo::execute(DDCompactView& cpv) {
  
  LogDebug("TIDGeom") << "==>> Constructing DDTIDModuleAlgo...";

  DDName parentName(parent().name()); 

  double sidfr = sideFrameWidth - sideFrameOver;      // width of side frame on the sides of module 
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
  double kaptonWidth = sidfr + kaptonOver;

  double  dxbot = 0.5*dlBottom + sidfr;
  double  dxtop = 0.5*dlTop + sidfr;
  double  dxtopenv, dxbotenv;           // top/bot width of the module envelope trap

  // Envelope 
  if (dlHybrid > dlTop) {
    // ring 1, ring 2
    dxtopenv = dxbot + (dxtop-dxbot)*(fullHeight+pitchHeight+topfr+hybridHeight)/fullHeight;
    dxbotenv = dxtop - (dxtop-dxbot)*(fullHeight+botfr)/fullHeight;
  } else {
    // ring 3
    dxtopenv = dxbot + (dxtop-dxbot)*(fullHeight+topfr)/fullHeight;
    dxbotenv = dxbot;
  }
  double bl1   = dxbotenv;
  double bl2   = dxtopenv;
  double h1    = 0.5 * moduleThick;
  double dz    = 0.5 * (boxFrameHeight + sideFrameHeight);

  DDSolid solidUncut, solidCut;
  DDSolid    solid = DDSolidFactory::trap(parentName, dz, 0, 0,
					  h1, bl1, bl1, 0, h1, bl2, bl2, 0);
  DDMaterial matter  = DDMaterial(DDName(DDSplit(genMat).first, DDSplit(genMat).second));
  DDLogicalPart module(solid.ddname(), matter, solid);
  LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
		      << " Trap made of " << genMat << " of dimensions " << dz 
		      << ", 0, 0, " << h1  << ", " << bl1 << ", " << bl1 
		      << ", 0, " << h1 << ", " << bl2  << ", " << bl2 
		      << ", 0";

  if (doComponents) {

    //Box frame
    matter  = DDMaterial(DDName(DDSplit(boxFrameMat).first, DDSplit(boxFrameMat).second));
    double dx = 0.5 * boxFrameWidth;
    double dy = 0.5 * boxFrameThick;
    double dz = 0.5 * boxFrameHeight; 
    solid = DDSolidFactory::box(DDName(DDSplit( boxFrameName ).first, DDSplit( boxFrameName ).second ),
				 dx, dy, dz);
    LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
			<< " Box made of " << matter.ddname() << " of dimensions " 
			<< dx << ", " << dy << ", " << dz;
    DDLogicalPart boxFrame(solid.ddname(), matter, solid);


    // Hybrid
    matter  = DDMaterial(DDName(DDSplit(hybridMat).first, DDSplit(hybridMat).second));
    dx = 0.5 * hybridWidth;
    dy = 0.5 * hybridThick;
    dz        = 0.5 * hybridHeight;
    solid = DDSolidFactory::box( DDName(DDSplit(hybridName).first, DDSplit(hybridName).second),
				 dx, dy, dz);
    LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
			<< " Box made of " << matter.ddname() << " of dimensions " 
			<< dx << ", " << dy << ", " << dz;
    DDLogicalPart hybrid(solid.ddname(), matter, solid);

    // Cool Insert
    matter  = DDMaterial(DDName(DDSplit(coolMat).first, DDSplit(coolMat).second));
    dx = 0.5 * coolWidth;
    dy = 0.5 * coolThick;
    dz        = 0.5 * coolHeight;
    solid = DDSolidFactory::box(DDName(DDSplit(coolName).first, DDSplit(coolName).second),
				dx, dy, dz);
    LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
			<< " Box made of " << matter.ddname() << " of dimensions " 
			<< dx << ", " << dy << ", " << dz;
    DDLogicalPart cool(solid.ddname(), matter, solid);

    // Loop over detectors to be placed
    for (int k = 0; k < detectorN; k++) {

      double bbl1, bbl2; // perhaps useless (bl1 enough)

      // Frame Sides
      matter  = DDMaterial(DDName(DDSplit(sideFrameMat).first,
				  DDSplit(sideFrameMat).second));
      if (dlHybrid > dlTop) {
	// ring 1, ring 2
	bbl1 = dxtop - (dxtop-dxbot)*(fullHeight+botfr)/fullHeight;
	bbl2 = dxbot + (dxtop-dxbot)*(fullHeight+pitchHeight+topfr)/fullHeight;
      } else {
	// ring 3
	bbl1 = dxtop - (dxtop-dxbot)*(fullHeight+pitchHeight+botfr)/fullHeight;
	bbl2 = dxbot + (dxtop-dxbot)*(fullHeight+topfr)/fullHeight;
      }
      h1 = 0.5 * sideFrameThick;
      dz = 0.5 * sideFrameHeight;
      solid = DDSolidFactory::trap(DDName(DDSplit(sideFrameName[k]).first,
					  DDSplit(sideFrameName[k]).second),
				   dz, 0, 0, h1, bbl1, bbl1, 0, 
				   h1,  bbl2, bbl2, 0);
      LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
			  << " Trap made of " << matter.ddname() << " of dimensions "
			  << dz << ", 0, 0, " << h1 << ", " << bbl1 << ", " 
			  << bbl1 << ", 0, " << h1 << ", " << bbl2 << ", " 
			  << bbl2 << ", 0";
      DDLogicalPart sideFrame(solid.ddname(), matter, solid);

      std::string rotstr, rotns; 
      DDRotation rot;

      // Hole in the frame below the wafer 
      matter  = DDMaterial(DDName(DDSplit(genMat).first, DDSplit(genMat).second));
      double xpos, zpos;
      dz        = fullHeight - bottomFrameOver - topFrameOver;
      bbl1     = dxbot - sideFrameWidth + bottomFrameOver*(dxtop-dxbot)/fullHeight;
      bbl2     = dxtop - sideFrameWidth - topFrameOver*(dxtop-dxbot)/fullHeight;
      if (dlHybrid > dlTop) {
	// ring 1, ring 2
	zpos    = -(topFrameHeight+0.5*dz-0.5*sideFrameHeight);
      } else {
	// ring 3
	zpos    = bottomFrameHeight+0.5*dz-0.5*sideFrameHeight;
      }
      dz     /= 2.;
      solid = DDSolidFactory::trap(DDName(DDSplit(holeFrameName[k]).first,
					  DDSplit(holeFrameName[k]).second),
				   dz, 0,0, h1,bbl1,bbl1, 0, 
				   h1,bbl2,bbl2, 0);
      LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
			  << " Trap made of " << matter.ddname() << " of dimensions "
			  << dz << ", 0, 0, " << h1 << ", " << bbl1 << ", " 
			  << bbl1 << ", 0, " << h1 << ", " << bbl2 << ", " 
			  << bbl2 << ", 0";
      DDLogicalPart holeFrame(solid.ddname(), matter, solid);      

      rotstr = DDSplit(holeFrameRot[k]).first;
      if (rotstr != "NULL") {
	rotns = DDSplit(holeFrameRot[k]).second;
	rot   = DDRotation(DDName(rotstr, rotns));      
      } else {
	rot     = DDRotation();
      }
      cpv.position(holeFrame, sideFrame, 1, DDTranslation(0.0, 0.0, zpos), rot );   
      LogDebug("TIDGeom") << "DDTIDModuleAlgo test: " << holeFrame.name() 
			  << " number 1 positioned in " << sideFrame.name()
			  << " at (0,0," << zpos << ") with no rotation";

      // Kapton circuit
      matter  = DDMaterial(DDName(DDSplit(kaptonMat).first,DDSplit(kaptonMat).second));
      double kaptonExtraHeight=0;      // kapton extra height in the stereo
      if (dlHybrid > dlTop) {
	// ring 1, ring 2
	bbl1 = dxtop - (dxtop-dxbot)*(fullHeight+botfr)/fullHeight;
	if ( k == 1 ) {
	  kaptonExtraHeight = dlTop*sin(detTilt)-fullHeight*(1-cos(detTilt));
	  kaptonExtraHeight = 0.5*fabs(kaptonExtraHeight);
	  bbl2 = dxbot + (dxtop-dxbot)*(fullHeight+kaptonExtraHeight)/fullHeight;
	}
        else {
	  bbl2 = dxtop;
	}
      } else {
	// ring 3
	bbl2 = dxbot + (dxtop-dxbot)*(fullHeight+topfr)/fullHeight;
	if ( k == 1) {
	  kaptonExtraHeight = dlBottom*sin(detTilt)-fullHeight*(1-cos(detTilt));
	  kaptonExtraHeight = 0.5*fabs(kaptonExtraHeight);
	  bbl1 = dxtop - (dxtop-dxbot)*(fullHeight+kaptonExtraHeight)/fullHeight;
	}  else {
	  bbl1 = dxbot;
	}
      }
      h1 = 0.5 * kaptonThick;
      dz = 0.5 * (kaptonHeight+kaptonExtraHeight);

      // For the stereo create the uncut solid, the solid to be removed and then the subtraction solid
      if ( k == 1 ) {
	// Uncut solid
	std::string kaptonUncutName=kaptonName[k]+"Uncut";
	solidUncut = DDSolidFactory::trap(DDName(DDSplit(kaptonUncutName).first,
						 DDSplit(kaptonUncutName).second),
					  dz, 0, 0, h1, bbl1, bbl1, 0,
					  h1,  bbl2, bbl2, 0);

	// Piece to be cut
	std::string kaptonCutName=kaptonName[k]+"Cut";

	if (dlHybrid > dlTop) {
	  dz   = 0.5 * dlTop;
	} else {
	  dz   = 0.5 * dlBottom;
	}
	h1      = 0.5 * kaptonThick;
	bbl1     =  fabs(dz*sin(detTilt));
	bbl2     =  bbl1*0.000001;
	double thet = atan((bbl1-bbl2)/(2*dz));	
	solidCut  = DDSolidFactory::trap(DDName(DDSplit(kaptonCutName).first,
						DDSplit(kaptonCutName).second),
					 dz, thet, 0, h1, bbl1, bbl1, 0,
					 h1, bbl2, bbl2, 0);

	std::string aRot("tidmodpar:9PYX"); 
	rotstr  = DDSplit(aRot).first;
	rotns = DDSplit(aRot).second;
	rot   = DDRotation(DDName(rotstr, rotns));

	xpos = -0.5 * fullHeight * sin(detTilt);
	zpos = 0.5 * kaptonHeight - bbl2;

	// Subtraction Solid
	solid  = DDSolidFactory::subtraction(DDName(DDSplit(kaptonName[k]).first,
						    DDSplit(kaptonName[k]).second),
					     solidUncut, solidCut, 
					     DDTranslation(xpos,0.0,zpos),rot);
      } else {
	solid  = DDSolidFactory::trap(DDName(DDSplit(kaptonName[k]).first,
					     DDSplit(kaptonName[k]).second),
				      dz, 0, 0, h1, bbl1, bbl1, 0, 
				      h1,  bbl2, bbl2, 0);
      }

      DDLogicalPart kapton(solid.ddname(), matter, solid);         
      LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
			  << " SUBTRACTION SOLID Trap made of " << matter.ddname() 
			  << " of dimensions " << dz << ", 0, 0, " << h1 
			  << ", " << bbl1 << ", " << bbl1 << ", 0, " << h1 
			  << ", " << bbl2 << ", " << bbl2 << ", 0";


      // Hole in the kapton below the wafer 
      matter  = DDMaterial(DDName(DDSplit(genMat).first, DDSplit(genMat).second));
      dz      = fullHeight - kaptonOver;
      xpos = 0; 
      if (dlHybrid > dlTop) {
	// ring 1, ring 2
	bbl1 = dxbot - kaptonWidth + kaptonOver*(dxtop-dxbot)/fullHeight;
	bbl2 = dxtop - kaptonWidth;
	zpos = 0.5*(kaptonHeight-kaptonExtraHeight-dz); 
	if ( k == 1 ) {
	  zpos -= 0.5*kaptonOver*(1-cos(detTilt));
	  xpos = -0.5*kaptonOver*sin(detTilt); 
	}
      } else {
	// ring 3
	bbl1 = dxbot - kaptonWidth;
	bbl2 = dxtop - kaptonWidth - kaptonOver*(dxtop-dxbot)/fullHeight;
	zpos = -0.5*(kaptonHeight-kaptonExtraHeight-dz);
      }
      dz     /= 2.;
      solid = DDSolidFactory::trap(DDName(DDSplit(holeKaptonName[k]).first,
					  DDSplit(holeKaptonName[k]).second),
				   dz, 0,0, h1,bbl1,bbl1, 0, 
				   h1,bbl2,bbl2, 0);
      LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
			  << " Trap made of " << matter.ddname() << " of dimensions "
			  << dz << ", 0, 0, " << h1 << ", " << bbl1 << ", " 
			  << bbl1 << ", 0, " << h1 << ", " << bbl2 << ", " 
			  << bbl2 << ", 0";
      DDLogicalPart holeKapton(solid.ddname(), matter, solid);      

      rotstr = DDSplit(holeKaptonRot[k]).first;
      if (rotstr != "NULL") {
       	rotns = DDSplit(holeKaptonRot[k]).second;
      	rot   = DDRotation(DDName(rotstr, rotns));      
      } else {
	rot     = DDRotation();
      }
      cpv.position(holeKapton, kapton, 1, DDTranslation(xpos, 0.0, zpos), rot );   
      LogDebug("TIDGeom") << "DDTIDModuleAlgo test: " << holeKapton.name() 
			  << " number 1 positioned in " << kapton.name()
			  << " at (0,0," << zpos << ") with no rotation";



      // Wafer
      matter  = DDMaterial(DDName(DDSplit(waferMat).first, DDSplit(waferMat).second));
      if (k == 0 && dlHybrid < dlTop) {
	bl1     = 0.5 * dlTop;
	bl2     = 0.5 * dlBottom;
      } else {
	bl1     = 0.5 * dlBottom;
	bl2     = 0.5 * dlTop;
      }
      h1      = 0.5 * waferThick[k];
      dz      = 0.5 * fullHeight;
      solid = DDSolidFactory::trap(DDName(DDSplit(waferName[k]).first,
					  DDSplit(waferName[k]).second),
				   dz, 0,0, h1,bl1,bl1,0, h1,bl2,bl2,0);
      LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
			  << " Trap made of " << matter.ddname() << " of dimensions "
			  << dz << ", 0, 0, " << h1 << ", " << bl1 << ", " 
			  << bl1 << ", 0, " << h1 << ", " << bl2 << ", "
			  << bl2 << ", 0";
      DDLogicalPart wafer(solid.ddname(), matter, solid);

      // Active
      matter  = DDMaterial(DDName(DDSplit(activeMat).first, DDSplit(activeMat).second));
      if (k == 0 && dlHybrid < dlTop) {
	bl1    -= sideWidthTop;
	bl2    -= sideWidthBottom;
      }
      else {
	bl1    -= sideWidthBottom;
	bl2    -= sideWidthTop;
      }
      dz      = 0.5 * (waferThick[k] - backplaneThick[k]); // inactive backplane
      h1      = 0.5 * activeHeight;
      solid = DDSolidFactory::trap(DDName(DDSplit(activeName[k]).first,
					  DDSplit(activeName[k]).second),
				   dz, 0,0, h1,bl2,bl1,0, h1,bl2,bl1,0);
      LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
			  << " Trap made of " << matter.ddname() << " of dimensions "
			  << dz << ", 0, 0, " << h1 << ", " << bl2 << ", " 
			  << bl1 << ", 0, " << h1 << ", " << bl2 << ", "
			  << bl1 << ", 0";
      DDLogicalPart active(solid.ddname(), matter, solid);
      rotstr = DDSplit(activeRot).first;
      if (rotstr != "NULL") {
	rotns = DDSplit(activeRot).second;
	rot   = DDRotation(DDName(rotstr, rotns));
      } else {
	rot     = DDRotation();
      }
      DDTranslation tran(0.0,-0.5 * backplaneThick[k],0.0); // from the definition of the wafer local axes
      cpv.position(active, wafer, 1, tran, rot);  // inactive backplane
      LogDebug("TIDGeom") << "DDTIDModuleAlgo test: " << active.name() 
			  << " number 1 positioned in " << wafer.name() 
			  << " at " << tran << " with " << rot;
      
      //Pitch Adapter
      matter  = DDMaterial(DDName(DDSplit(pitchMat).first, DDSplit(pitchMat).second));
      if (dlHybrid > dlTop) {
	dz   = 0.5 * dlTop;
      } else {
	dz   = 0.5 * dlBottom;
      }
      if (k == 0) {
	dx      = dz;
	dy      = 0.5 * pitchThick;
	dz      = 0.5 * pitchHeight;
	solid   = DDSolidFactory::box(DDName(DDSplit(pitchName[k]).first,
					     DDSplit(pitchName[k]).second),
				      dx, dy, dz);
	LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name()
			    << " Box made of " << matter.ddname() << " of dimensions"
			    << " " << dx << ", " << dy << ", " << dz;
      } else {
	h1      = 0.5 * pitchThick;
	bl1     = 0.5 * pitchHeight + 0.5 * dz * sin(detTilt);
	bl2     = 0.5 * pitchHeight - 0.5 * dz * sin(detTilt);

	dz -=0.5*pitchStereoTol;
	bl1-=pitchStereoTol;
	bl2-=pitchStereoTol;

	double thet = atan((bl1-bl2)/(2.*dz));
	solid   = DDSolidFactory::trap(DDName(DDSplit(pitchName[k]).first,
					     DDSplit(pitchName[k]).second),
				       dz, thet, 0, h1, bl1, bl1, 0, 
				       h1, bl2, bl2, 0);
	LogDebug("TIDGeom") << "DDTIDModuleAlgo test:\t" << solid.name() 
			    << " Trap made of " << matter.ddname() << " of "
			    << "dimensions " << dz << ", " << thet/CLHEP::deg 
			    << ", 0, " << h1 << ", " << bl1 << ", " << bl1 
			    << ", 0, " << h1 << ", " << bl2 << ", " << bl2 
			    << ", 0";
      }
      DDLogicalPart pa(solid.ddname(), matter, solid);
    }
  }
  LogDebug("TIDGeom") << "<<== End of DDTIDModuleAlgo construction ...";
}
