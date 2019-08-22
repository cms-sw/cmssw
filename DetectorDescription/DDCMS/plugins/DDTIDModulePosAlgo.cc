#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace dd4hep;
using namespace cms;
using namespace cms_units::operators;

static long algorithm(Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e, SensitiveDetector& /* sens */) {
  cms::DDNamespace ns(ctxt, e, true);
  DDAlgoArguments args(ctxt, e);
  string parentName = args.parentName();
  int detectorN = args.integer("DetectorNumber");             //Number of detectors
  double detTilt = args.dble("DetTilt");                      //Tilt of stereo detector
  double fullHeight = args.dble("FullHeight");                //Height
  string boxFrameName = args.str("BoxFrameName");             //Top frame Name
  double boxFrameHeight = args.dble("BoxFrameHeight");        //          height
  double boxFrameWidth = args.dble("BoxFrameWidth");          //          width
  double dlTop = args.dble("DlTop");                          //Width at top of wafer
  double dlBottom = args.dble("DlBottom");                    //Width at bottom of wafer
  double dlHybrid = args.dble("DlHybrid");                    //Width at the hybrid end
  vector<double> boxFrameZ = args.vecDble("BoxFrameZ");       //              z-positions
  double bottomFrameHeight = args.dble("BottomFrameHeight");  //Bottom of the frame
  double bottomFrameOver = args.dble("BottomFrameOver");      //              overlap
  double topFrameHeight = args.dble("TopFrameHeight");        //Top    of the frame
  double topFrameOver = args.dble("TopFrameOver");            //              overlap

  vector<string> sideFrameName = args.vecStr("SideFrameName");  //Side Frame    name
  vector<double> sideFrameZ = args.vecDble("SideFrameZ");       //              z-positions
  vector<string> sideFrameRot = args.vecStr(
      "SideFrameRotation");  //              rotation matrix (required for correct positiong of the hole in the StereoR)
  double sideFrameWidth = args.dble("SideFrameWidth");  //              width
  double sideFrameOver = args.dble("SideFrameOver");    //              overlap (wrt wafer)

  vector<string> kaptonName = args.vecStr("KaptonName");  //Kapton Circuit    name
  vector<double> kaptonZ = args.vecDble("KaptonZ");       //              z-positions
  vector<string> kaptonRot = args.vecStr(
      "KaptonRotation");  //              rotation matrix (required for correct positiong of the hole in the StereoR)
  vector<string> waferName = args.vecStr("WaferName");            //Wafer         name
  vector<double> waferZ = args.vecDble("WaferZ");                 //              z-positions
  vector<string> waferRot = args.vecStr("WaferRotation");         //              rotation matrix
  string hybridName = args.str("HybridName");                     //Hybrid        name
  double hybridHeight = args.dble("HybridHeight");                //              height
  vector<double> hybridZ = args.vecDble("HybridZ");               //              z-positions
  vector<string> pitchName = args.vecStr("PitchName");            //Pitch adapter rotation matrix
  double pitchHeight = args.dble("PitchHeight");                  //              height
  vector<double> pitchZ = args.vecDble("PitchZ");                 //              z-positions
  vector<string> pitchRot = args.vecStr("PitchRotation");         //              rotation matrix
  string coolName = args.str("CoolInsertName");                   //Cool Insert   name
  double coolHeight = args.dble("CoolInsertHeight");              //              height
  double coolZ = args.dble("CoolInsertZ");                        //              z-position
  double coolWidth = args.dble("CoolInsertWidth");                //              width
  vector<double> coolRadShift = args.vecDble("CoolInsertShift");  //

  bool doSpacers =
      ::toupper(args.str("DoSpacers")[0]) != 'N';  //Spacers (alumina) to be made (Should be "Yes" for DS modules only)
  string botSpacersName = args.str("BottomSpacersName");       // Spacers at the "bottom" of the module
  double botSpacersHeight = args.dble("BottomSpacersHeight");  //
  double botSpacersZ = args.dble("BottomSpacersZ");            //              z-position
  string sidSpacersName = args.str("SideSpacersName");         //Spacers at the "sides" of the module
  double sidSpacersHeight = args.dble("SideSpacersHeight");
  double sidSpacersZ = args.dble("SideSpacersZ");             //              z-position
  double sidSpacersWidth = args.dble("SideSpacersWidth");     //              width
  double sidSpacersRadShift = args.dble("SideSpacersShift");  //

  LogDebug("TIDGeom") << "Parent " << parentName << " Detector Planes " << detectorN;
  LogDebug("TIDGeom") << "Detector Tilt " << convertRadToDeg(detTilt) << " Height " << fullHeight << " dl(Top) "
                      << dlTop << " dl(Bottom) " << dlBottom << " dl(Hybrid) " << dlHybrid;
  LogDebug("TIDGeom") << boxFrameName << " positioned at Z";
  for (int i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "\tboxFrameZ[" << i << "] = " << boxFrameZ[i];
  LogDebug("TIDGeom") << "\t Extra Height at Bottom " << bottomFrameHeight << " Overlap " << bottomFrameOver;
  for (int i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "\tsideFrame[" << i << "] = " << sideFrameName[i] << " positioned at Z " << sideFrameZ[i]
                        << " with rotation " << sideFrameRot[i];
  for (int i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "\tkapton[" << i << "] = " << kaptonName[i] << " positioned at Z " << kaptonZ[i]
                        << " with rotation " << kaptonRot[i];
  for (int i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << waferName[i] << " positioned at Z " << waferZ[i] << " with rotation " << waferRot[i];
  LogDebug("TIDGeom") << hybridName << " Height " << hybridHeight << " Z";
  for (int i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "\thybridZ[" << i << "] = " << hybridZ[i];
  LogDebug("TIDGeom") << "Pitch Adapter Height " << pitchHeight;
  for (int i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << pitchName[i] << " position at Z " << pitchZ[i] << " with rotation " << pitchRot[i];

  string name;
  double botfr;  // width of side frame at the the bottom of the modules
  double topfr;  // width of side frame at the the top of the modules
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
  double zCenter = 0.5 * (sideFrameHeight + boxFrameHeight);

  // (Re) Compute the envelope for positioning Cool Inserts and Side Spacers (Alumina).
  double sidfr = sideFrameWidth - sideFrameOver;  // width of side frame on the sides of module
  double dxbot = 0.5 * dlBottom + sidfr;
  double dxtop = 0.5 * dlTop + sidfr;
  double dxtopenv, dxbotenv;  // top/bot width of the module envelope trap

  double tanWafer = (dxtop - dxbot) / fullHeight;  //
  double thetaWafer = atan(tanWafer);              // 1/2 of the wafer wedge angle

  if (dlHybrid > dlTop) {
    // ring 1, ring 2
    dxtopenv = dxbot + (dxtop - dxbot) * (fullHeight + pitchHeight + topfr + hybridHeight) / fullHeight;
    dxbotenv = dxtop - (dxtop - dxbot) * (fullHeight + botfr) / fullHeight;
  } else {
    // ring 3
    dxtopenv = dxbot + (dxtop - dxbot) * (fullHeight + topfr) / fullHeight;
    dxbotenv = dxbot;
  }

  double tanEnv = (dxtopenv - dxbotenv) / (sideFrameHeight + boxFrameHeight);  // 1/2 of the envelope wedge angle

  double xpos = 0;
  double ypos = 0;
  double zpos = 0;

  // Cool Inserts
  name = coolName;
  ypos = coolZ;

  double zCool;
  int copy = 0;
  Rotation3D rot;  // should be different for different elements
  Volume parentVol = ns.volume(parentName);

  for (int j1 = 0; j1 < 2; j1++) {  // j1: 0 inserts below the hybrid
    //     1 inserts below the wafer
    if (dlHybrid > dlTop) {
      zCool = sideFrameHeight + boxFrameHeight - coolRadShift[j1];
      if (j1 == 0)
        zCool -= 0.5 * coolHeight;
    } else {
      zCool = coolRadShift[j1];
      if (j1 == 0)
        zCool += 0.5 * coolHeight;
    }

    if (j1 == 0) {
      xpos = -0.5 * (boxFrameWidth - coolWidth);
    } else {
      xpos = -(dxbotenv + (zCool - 0.5 * coolHeight) * tanEnv - 0.5 * coolWidth);
    }

    zpos = zCool - zCenter;
    for (int j2 = 0; j2 < 2; j2++) {
      copy++;
      parentVol.placeVolume(ns.volume(name), copy, Position(xpos, ypos, zpos));
      LogDebug("TIDGeom") << name << " number " << copy << " positioned in " << parentName << " at "
                          << Position(xpos, ypos, zpos) << " with " << rot;
      xpos = -xpos;
    }
  }

  if (doSpacers) {
    // Bottom Spacers (Alumina)
    name = botSpacersName;
    ypos = botSpacersZ;
    double zBotSpacers;
    if (dlHybrid > dlTop) {
      zBotSpacers = sideFrameHeight + boxFrameHeight - 0.5 * botSpacersHeight;
    } else {
      zBotSpacers = 0.5 * botSpacersHeight;
    }
    zpos = zBotSpacers - zCenter;
    parentVol.placeVolume(ns.volume(name), 1, Position(0.0, ypos, zpos));
    LogDebug("TIDGeom") << name << " number " << 1 << " positioned in " << parentName << " at "
                        << Position(0.0, ypos, zpos) << " with no rotation";
    // Side Spacers (Alumina)
    name = sidSpacersName;
    ypos = sidSpacersZ;
    double zSideSpacers;
    if (dlHybrid > dlTop) {
      zSideSpacers = sideFrameHeight + boxFrameHeight - sidSpacersRadShift;
    } else {
      zSideSpacers = sidSpacersRadShift;
    }
    zpos = zSideSpacers - zCenter;

    copy = 0;
    xpos = dxbotenv + (zSideSpacers - 0.5 * sidSpacersHeight) * tanEnv - 0.5 * sidSpacersWidth + sideFrameOver;

    double phiy = 0e0, phiz = 0e0;
    double phix = 0._deg;
    phiy = 90._deg;
    phiz = 0._deg;

    double thetax = 0e0;
    double thetay = 90._deg;
    double thetaz = thetaWafer;

    for (int j1 = 0; j1 < 2; j1++) {
      copy++;
      // tilt Side Spacers (parallel to Side Frame)
      thetax = 90._deg + thetaz;
      rot = makeRotation3D(thetax, phix, thetay, phiy, thetaz, phiz);
      parentVol.placeVolume(ns.volume(name), copy, Transform3D(rot, Position(xpos, ypos, zpos)));
      LogDebug("TIDGeom") << name << " number " << copy << " positioned in " << parentName << " at "
                          << Position(xpos, ypos, zpos) << " with " << rot;
      xpos = -xpos;
      thetaz = -thetaz;
    }
  }

  // Loop over detectors to be placed
  for (int k = 0; k < detectorN; k++) {
    // Wafer
    name = waferName[k];
    xpos = 0;
    ypos = waferZ[k];
    double zWafer;
    if (dlHybrid > dlTop) {
      zWafer = botfr + 0.5 * fullHeight;
    } else {
      zWafer = boxFrameHeight + botfr + pitchHeight + 0.5 * fullHeight;
    }
    zpos = zWafer - zCenter;
    Position tran(xpos, ypos, zpos);
    rot = ns.rotation(waferRot[k]);

    parentVol.placeVolume(ns.volume(name), k + 1, Transform3D(rot, tran));  // copyNr=k+1
    LogDebug("TIDGeom") << name << " number " << k + 1 << " positioned in " << parentName << " at " << tran << " with "
                        << rot;

    //Pitch Adapter
    name = pitchName[k];
    if (k == 0) {
      xpos = 0;
    } else {
      xpos = 0.5 * fullHeight * sin(detTilt);
    }
    ypos = pitchZ[k];
    double zPitch;
    if (dlHybrid > dlTop) {
      zPitch = botfr + fullHeight + 0.5 * pitchHeight;
    } else {
      zPitch = boxFrameHeight + botfr + 0.5 * pitchHeight;
    }
    zpos = zPitch - zCenter;
    rot = ns.rotation(pitchRot[k]);
    tran = Position(xpos, ypos, zpos);
    parentVol.placeVolume(ns.volume(name), k + 1, Transform3D(rot, tran));  // copyNr=k+1
    LogDebug("TIDGeom") << name << " number " << k + 1 << " positioned in " << parentName << " at " << tran << " with "
                        << rot;

    // Hybrid
    name = hybridName;
    ypos = hybridZ[k];
    double zHybrid;
    if (dlHybrid > dlTop) {
      zHybrid = botfr + fullHeight + pitchHeight + 0.5 * hybridHeight;
    } else {
      zHybrid = 0.5 * hybridHeight;
    }
    zpos = zHybrid - zCenter;
    tran = Position(0, ypos, zpos);
    parentVol.placeVolume(ns.volume(name), k + 1, tran);  // copyNr=k+1
    LogDebug("TIDGeom") << name << " number " << k + 1 << " positioned in " << parentName << " at " << tran;

    // Box frame
    name = boxFrameName;
    ypos = boxFrameZ[k];
    double zBoxFrame;
    if (dlHybrid > dlTop) {
      zBoxFrame = sideFrameHeight + 0.5 * boxFrameHeight;
    } else {
      zBoxFrame = 0.5 * boxFrameHeight;
    }
    zpos = zBoxFrame - zCenter;
    tran = Position(0, ypos, zpos);
    parentVol.placeVolume(ns.volume(name), k + 1, tran);  // copyNr=k+1
    LogDebug("TIDGeom") << name << " number " << k + 1 << " positioned in " << parentName << " at " << tran;

    // Side frame
    name = sideFrameName[k];
    ypos = sideFrameZ[k];
    double zSideFrame;
    if (dlHybrid > dlTop) {
      zSideFrame = 0.5 * sideFrameHeight;
    } else {
      zSideFrame = boxFrameHeight + 0.5 * sideFrameHeight;
    }
    zpos = zSideFrame - zCenter;
    rot = ns.rotation(sideFrameRot[k]);
    tran = Position(0, ypos, zpos);
    parentVol.placeVolume(ns.volume(name), k + 1, Transform3D(rot, tran));
    LogDebug("TIDGeom") << name << " number " << k + 1 << " positioned in " << parentName << " at " << tran << " with "
                        << rot;
    // Kapton circuit
    name = kaptonName[k];
    ypos = kaptonZ[k];
    double zKapton;
    double kaptonExtraHeight = 0;
    if (dlHybrid > dlTop) {
      if (k == 1)
        kaptonExtraHeight = dlTop * sin(detTilt) - fullHeight * (1 - cos(detTilt));
      kaptonExtraHeight = 0.5 * fabs(kaptonExtraHeight);
      zKapton = 0.5 * (kaptonHeight + kaptonExtraHeight);
    } else {
      if (k == 1)
        kaptonExtraHeight = dlBottom * sin(detTilt) - fullHeight * (1 - cos(detTilt));
      kaptonExtraHeight = 0.5 * fabs(kaptonExtraHeight);
      zKapton = boxFrameHeight + sideFrameHeight - 0.5 * (kaptonHeight + kaptonExtraHeight);
    }
    zpos = zKapton - zCenter;
    rot = ns.rotation(kaptonRot[k]);
    tran = Position(0, ypos, zpos);
    parentVol.placeVolume(ns.volume(name), k + 1, Transform3D(rot, tran));
    LogDebug("TIDGeom") << name << " number " << k + 1 << " positioned in " << parentName << " at " << tran << " with "
                        << rot;
  }
  LogDebug("TIDGeom") << "<<== End of DDTIDModulePosAlgo positioning ...";
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTIDModulePosAlgo, algorithm)
