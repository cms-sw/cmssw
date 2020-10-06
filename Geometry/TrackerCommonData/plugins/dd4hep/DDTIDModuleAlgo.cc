#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace dd4hep;
using namespace cms;
using namespace cms_units::operators;

static long algorithm(Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  DDAlgoArguments args(ctxt, e);
  string mother = args.parentName();
  string genMat = args.str("GeneralMaterial");                        //General material name
  int detectorN = args.integer("DetectorNumber");                     //Detector planes
  double moduleThick = args.dble("ModuleThick");                      //Module thickness
  double detTilt = args.dble("DetTilt");                              //Tilt of stereo detector
  double fullHeight = args.dble("FullHeight");                        //Height
  double dlTop = args.dble("DlTop");                                  //Width at top of wafer
  double dlBottom = args.dble("DlBottom");                            //Width at bottom of wafer
  double dlHybrid = args.dble("DlHybrid");                            //Width at the hybrid end
  bool doComponents = ::toupper(args.str("DoComponents")[0]) != 'N';  //Components to be made

  string boxFrameName = args.str("BoxFrameName");                  //Top frame     name
  string boxFrameMat = args.str("BoxFrameMaterial");               //              material
  double boxFrameHeight = args.dble("BoxFrameHeight");             //              height
  double boxFrameThick = args.dble("BoxFrameThick");               //              thickness
  double boxFrameWidth = args.dble("BoxFrameWidth");               //              extra width
  double bottomFrameHeight = args.dble("BottomFrameHeight");       //Bottom of the frame
  double bottomFrameOver = args.dble("BottomFrameOver");           //    overlap
  double topFrameHeight = args.dble("TopFrameHeight");             //Top    of the frame
  double topFrameOver = args.dble("TopFrameOver");                 //              overlap
  vector<string> sideFrameName = args.vecStr("SideFrameName");     //Side frame    name
  string sideFrameMat = args.str("SideFrameMaterial");             //              material
  double sideFrameWidth = args.dble("SideFrameWidth");             //              width
  double sideFrameThick = args.dble("SideFrameThick");             //              thickness
  double sideFrameOver = args.dble("SideFrameOver");               //              overlap (wrt wafer)
  vector<string> holeFrameName = args.vecStr("HoleFrameName");     //Hole in the frame   name
  vector<string> holeFrameRot = args.vecStr("HoleFrameRotation");  //            Rotation matrix

  vector<string> kaptonName = args.vecStr("KaptonName");             //Kapton circuit name
  string kaptonMat = args.str("KaptonMaterial");                     //               material
  double kaptonThick = args.dble("KaptonThick");                     //               thickness
  double kaptonOver = args.dble("KaptonOver");                       //               overlap (wrt Wafer)
  vector<string> holeKaptonName = args.vecStr("HoleKaptonName");     //Hole in the kapton circuit name
  vector<string> holeKaptonRot = args.vecStr("HoleKaptonRotation");  //           Rotation matrix

  vector<string> waferName = args.vecStr("WaferName");    //Wafer         name
  string waferMat = args.str("WaferMaterial");            //              material
  double sideWidthTop = args.dble("SideWidthTop");        //              width on the side Top
  double sideWidthBottom = args.dble("SideWidthBottom");  //                                Bottom
  vector<string> activeName = args.vecStr("ActiveName");  //Sensitive     name
  string activeMat = args.str("ActiveMaterial");          //              material
  double activeHeight = args.dble("ActiveHeight");        //              height
  vector<double> waferThick =
      args.vecDble("WaferThick");                 //              wafer thickness (active       = wafer - backplane)
  string activeRot = args.str("ActiveRotation");  //              Rotation matrix
  vector<double> backplaneThick = args.vecDble("BackPlaneThick");  //              thickness
  string hybridName = args.str("HybridName");                      //Hybrid        name
  string hybridMat = args.str("HybridMaterial");                   //              material
  double hybridHeight = args.dble("HybridHeight");                 //              height
  double hybridWidth = args.dble("HybridWidth");                   //              width
  double hybridThick = args.dble("HybridThick");                   //              thickness
  vector<string> pitchName = args.vecStr("PitchName");             //Pitch adapter name
  string pitchMat = args.str("PitchMaterial");                     //              material
  double pitchHeight = args.dble("PitchHeight");                   //              height
  double pitchThick = args.dble("PitchThick");                     //              thickness
  double pitchStereoTol = args.dble("PitchStereoTolerance");       //           tolerance in dimensions of the stereo
  string coolName = args.str("CoolInsertName");                    // Cool insert name
  string coolMat = args.str("CoolInsertMaterial");                 //              material
  double coolHeight = args.dble("CoolInsertHeight");               //              height
  double coolThick = args.dble("CoolInsertThick");                 //              thickness
  double coolWidth = args.dble("CoolInsertWidth");                 //              width

  LogDebug("TIDGeom") << "Parent " << mother << " General Material " << genMat << " Detector Planes " << detectorN;

  LogDebug("TIDGeom") << "ModuleThick " << moduleThick << " Detector Tilt " << convertRadToDeg(detTilt) << " Height "
                      << fullHeight << " dl(Top) " << dlTop << " dl(Bottom) " << dlBottom << " dl(Hybrid) " << dlHybrid
                      << " doComponents " << doComponents;
  LogDebug("TIDGeom") << "" << boxFrameName << " Material " << boxFrameMat << " Thickness " << boxFrameThick
                      << " width " << boxFrameWidth << " height " << boxFrameHeight << " Extra Height at Bottom "
                      << bottomFrameHeight << " Overlap " << bottomFrameOver;

  for (int i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << sideFrameName[i] << " Material " << sideFrameMat << " Width " << sideFrameWidth
                        << " Thickness " << sideFrameThick << " Overlap " << sideFrameOver << " Hole  "
                        << holeFrameName[i];

  for (int i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << kaptonName[i] << " Material " << kaptonMat << " Thickness " << kaptonThick << " Overlap "
                        << kaptonOver << " Hole  " << holeKaptonName[i];

  LogDebug("TIDGeom") << "Wafer Material " << waferMat << " Side Width Top " << sideWidthTop << " Side Width Bottom "
                      << sideWidthBottom;
  for (int i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "\twaferName[" << i << "] = " << waferName[i];

  LogDebug("TIDGeom") << "Active Material " << activeMat << " Height " << activeHeight << " rotated by " << activeRot;
  for (int i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << " translated by (0," << -0.5 * backplaneThick[i] << ",0)\tactiveName[" << i
                        << "] = " << activeName[i] << " of thickness " << waferThick[i] - backplaneThick[i];

  LogDebug("TIDGeom") << "" << hybridName << " Material " << hybridMat << " Height " << hybridHeight << " Width "
                      << hybridWidth << " Thickness " << hybridThick;
  LogDebug("TIDGeom") << "Pitch Adapter Material " << pitchMat << " Height " << pitchHeight << " Thickness "
                      << pitchThick;
  for (int i = 0; i < detectorN; i++)
    LogDebug("TIDGeom") << "\tpitchName[" << i << "] = " << pitchName[i];
  LogDebug("TIDGeom") << "Cool Element Material " << coolMat << " Height " << coolHeight << " Thickness " << coolThick
                      << " Width " << coolWidth;

  string name = mother;
  double sidfr = sideFrameWidth - sideFrameOver;  // width of side frame on the sides of module
  double botfr;                                   // width of side frame at the the bottom of the modules
  double topfr;                                   // width of side frame at the the top of the modules
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

  double dxbot = 0.5 * dlBottom + sidfr;
  double dxtop = 0.5 * dlTop + sidfr;
  double dxtopenv, dxbotenv;  // top/bot width of the module envelope trap

  // Envelope
  if (dlHybrid > dlTop) {
    // ring 1, ring 2
    dxtopenv = dxbot + (dxtop - dxbot) * (fullHeight + pitchHeight + topfr + hybridHeight) / fullHeight;
    dxbotenv = dxtop - (dxtop - dxbot) * (fullHeight + botfr) / fullHeight;
  } else {
    // ring 3
    dxtopenv = dxbot + (dxtop - dxbot) * (fullHeight + topfr) / fullHeight;
    dxbotenv = dxbot;
  }
  double bl1 = dxbotenv;
  double bl2 = dxtopenv;
  double h1 = 0.5 * moduleThick;
  double dx, dy;
  double dz = 0.5 * (boxFrameHeight + sideFrameHeight);

  Solid solid = ns.addSolidNS(name, Trap(dz, 0, 0, h1, bl1, bl1, 0, h1, bl2, bl2, 0));
  /* Volume module = */ ns.addVolumeNS(Volume(name, solid, ns.material(genMat)));
  LogDebug("TIDGeom") << solid.name() << " Trap made of " << genMat << " of dimensions " << dz << ", 0, 0, " << h1
                      << ", " << bl1 << ", " << bl1 << ", 0, " << h1 << ", " << bl2 << ", " << bl2 << ", 0";

  if (doComponents) {
    //Box frame
    name = boxFrameName;
    dx = 0.5 * boxFrameWidth;
    dy = 0.5 * boxFrameThick;
    dz = 0.5 * boxFrameHeight;
    solid = ns.addSolidNS(name, Box(dx, dy, dz));
    LogDebug("TIDGeom") << solid.name() << " Box made of " << boxFrameMat << " of dimensions " << dx << ", " << dy
                        << ", " << dz;
    /* Volume boxFrame = */ ns.addVolumeNS(Volume(name, solid, ns.material(boxFrameMat)));

    // Hybrid
    name = hybridName;
    dx = 0.5 * hybridWidth;
    dy = 0.5 * hybridThick;
    dz = 0.5 * hybridHeight;
    solid = ns.addSolidNS(name, Box(dx, dy, dz));
    LogDebug("TIDGeom") << solid.name() << " Box made of " << hybridMat << " of dimensions " << dx << ", " << dy << ", "
                        << dz;
    /* Volume hybrid = */ ns.addVolumeNS(Volume(name, solid, ns.material(hybridMat)));

    // Cool Insert
    name = coolName;
    dx = 0.5 * coolWidth;
    dy = 0.5 * coolThick;
    dz = 0.5 * coolHeight;
    solid = ns.addSolidNS(name, Box(dx, dy, dz));
    LogDebug("TIDGeom") << solid.name() << " Box made of " << coolMat << " of dimensions " << dx << ", " << dy << ", "
                        << dz;
    /* Volume cool = */ ns.addVolumeNS(Volume(name, solid, ns.material(coolMat)));

    // Loop over detectors to be placed
    for (int k = 0; k < detectorN; k++) {
      double bbl1, bbl2;  // perhaps useless (bl1 enough)
      // Frame Sides
      name = sideFrameName[k];
      if (dlHybrid > dlTop) {
        // ring 1, ring 2
        bbl1 = dxtop - (dxtop - dxbot) * (fullHeight + botfr) / fullHeight;
        bbl2 = dxbot + (dxtop - dxbot) * (fullHeight + pitchHeight + topfr) / fullHeight;
      } else {
        // ring 3
        bbl1 = dxtop - (dxtop - dxbot) * (fullHeight + pitchHeight + botfr) / fullHeight;
        bbl2 = dxbot + (dxtop - dxbot) * (fullHeight + topfr) / fullHeight;
      }
      h1 = 0.5 * sideFrameThick;
      dz = 0.5 * sideFrameHeight;
      solid = ns.addSolidNS(name, Trap(dz, 0., 0., h1, bbl1, bbl1, 0., h1, bbl2, bbl2, 0.));
      LogDebug("TIDGeom") << solid.name() << " Trap made of " << sideFrameMat << " of dimensions " << dz << ", 0, 0, "
                          << h1 << ", " << bbl1 << ", " << bbl1 << ", 0, " << h1 << ", " << bbl2 << ", " << bbl2
                          << ", 0";
      Volume sideFrame = ns.addVolumeNS(Volume(name, solid, ns.material(sideFrameMat)));

      std::string rotstr, rotns;
      Rotation3D rot;

      // Hole in the frame below the wafer
      name = holeFrameName[k];
      double xpos, zpos;
      dz = fullHeight - bottomFrameOver - topFrameOver;
      bbl1 = dxbot - sideFrameWidth + bottomFrameOver * (dxtop - dxbot) / fullHeight;
      bbl2 = dxtop - sideFrameWidth - topFrameOver * (dxtop - dxbot) / fullHeight;
      if (dlHybrid > dlTop) {
        // ring 1, ring 2
        zpos = -(topFrameHeight + 0.5 * dz - 0.5 * sideFrameHeight);
      } else {
        // ring 3
        zpos = bottomFrameHeight + 0.5 * dz - 0.5 * sideFrameHeight;
      }
      dz /= 2.;
      solid = ns.addSolidNS(name, Trap(dz, 0, 0, h1, bbl1, bbl1, 0, h1, bbl2, bbl2, 0));
      LogDebug("TIDGeom") << solid.name() << " Trap made of " << genMat << " of dimensions " << dz << ", 0, 0, " << h1
                          << ", " << bbl1 << ", " << bbl1 << ", 0, " << h1 << ", " << bbl2 << ", " << bbl2 << ", 0";
      Volume holeFrame = ns.addVolumeNS(Volume(name, solid, ns.material(genMat)));

      rot = ns.rotation(holeFrameRot[k]);
      sideFrame.placeVolume(holeFrame, 1, Transform3D(rot, Position(0e0, 0e0, zpos)));  // copyNr=1
      LogDebug("TIDGeom") << holeFrame.name() << " number 1 positioned in " << sideFrame.name() << " at (0,0," << zpos
                          << ") with no rotation";

      // Kapton circuit
      double kaptonExtraHeight = 0;  // kapton extra height in the stereo
      if (dlHybrid > dlTop) {
        // ring 1, ring 2
        bbl1 = dxtop - (dxtop - dxbot) * (fullHeight + botfr) / fullHeight;
        if (k == 1) {
          kaptonExtraHeight = dlTop * sin(detTilt) - fullHeight * (1 - cos(detTilt));
          kaptonExtraHeight = 0.5 * fabs(kaptonExtraHeight);
          bbl2 = dxbot + (dxtop - dxbot) * (fullHeight + kaptonExtraHeight) / fullHeight;
        } else {
          bbl2 = dxtop;
        }
      } else {
        // ring 3
        bbl2 = dxbot + (dxtop - dxbot) * (fullHeight + topfr) / fullHeight;
        if (k == 1) {
          kaptonExtraHeight = dlBottom * sin(detTilt) - fullHeight * (1 - cos(detTilt));
          kaptonExtraHeight = 0.5 * fabs(kaptonExtraHeight);
          bbl1 = dxtop - (dxtop - dxbot) * (fullHeight + kaptonExtraHeight) / fullHeight;
        } else {
          bbl1 = dxbot;
        }
      }
      h1 = 0.5 * kaptonThick;
      dz = 0.5 * (kaptonHeight + kaptonExtraHeight);

      // For the stereo create the uncut solid, the solid to be removed and then the subtraction solid
      if (k == 1) {
        // Uncut solid
        Solid solidUncut =
            ns.addSolidNS(kaptonName[k] + "Uncut", Trap(dz, 0., 0., h1, bbl1, bbl1, 0., h1, bbl2, bbl2, 0));

        // Piece to be cut
        dz = (dlHybrid > dlTop) ? 0.5 * dlTop : 0.5 * dlBottom;
        h1 = 0.5 * kaptonThick;
        bbl1 = fabs(dz * sin(detTilt));
        bbl2 = bbl1 * 0.000001;
        double thet = atan((bbl1 - bbl2) / (2 * dz));
        Solid solidCut =
            ns.addSolidNS(kaptonName[k] + "Cut", Trap(dz, thet, 0., h1, bbl1, bbl1, 0., h1, bbl2, bbl2, 0));

        // Subtraction Solid
        name = kaptonName[k];
        rot = ns.rotation("tidmodpar:9PYX");
        xpos = -0.5 * fullHeight * sin(detTilt);
        zpos = 0.5 * kaptonHeight - bbl2;
        solid =
            ns.addSolidNS(name, SubtractionSolid(solidUncut, solidCut, Transform3D(rot, Position(xpos, 0.0, zpos))));
      } else {
        name = kaptonName[k];
        solid = ns.addSolidNS(name, Trap(dz, 0., 0., h1, bbl1, bbl1, 0., h1, bbl2, bbl2, 0.));
      }

      Volume kapton = ns.addVolumeNS(Volume(name, solid, ns.material(kaptonMat)));
      LogDebug("TIDGeom") << solid.name() << " SUBTRACTION SOLID Trap made of " << kaptonMat << " of dimensions " << dz
                          << ", 0, 0, " << h1 << ", " << bbl1 << ", " << bbl1 << ", 0, " << h1 << ", " << bbl2 << ", "
                          << bbl2 << ", 0";

      // Hole in the kapton below the wafer
      name = holeKaptonName[k];
      dz = fullHeight - kaptonOver;
      xpos = 0;
      if (dlHybrid > dlTop) {
        // ring 1, ring 2
        bbl1 = dxbot - kaptonWidth + kaptonOver * (dxtop - dxbot) / fullHeight;
        bbl2 = dxtop - kaptonWidth;
        zpos = 0.5 * (kaptonHeight - kaptonExtraHeight - dz);
        if (k == 1) {
          zpos -= 0.5 * kaptonOver * (1 - cos(detTilt));
          xpos = -0.5 * kaptonOver * sin(detTilt);
        }
      } else {
        // ring 3
        bbl1 = dxbot - kaptonWidth;
        bbl2 = dxtop - kaptonWidth - kaptonOver * (dxtop - dxbot) / fullHeight;
        zpos = -0.5 * (kaptonHeight - kaptonExtraHeight - dz);
      }
      dz /= 2.;
      solid = ns.addSolidNS(name, Trap(dz, 0., 0., h1, bbl1, bbl1, 0., h1, bbl2, bbl2, 0.));
      LogDebug("TIDGeom") << solid.name() << " Trap made of " << genMat << " of dimensions " << dz << ", 0, 0, " << h1
                          << ", " << bbl1 << ", " << bbl1 << ", 0, " << h1 << ", " << bbl2 << ", " << bbl2 << ", 0";
      Volume holeKapton = ns.addVolumeNS(Volume(name, solid, ns.material(genMat)));

      rot = ns.rotation(holeKaptonRot[k]);
      kapton.placeVolume(holeKapton, 1, Transform3D(rot, Position(xpos, 0.0, zpos)));
      LogDebug("TIDGeom") << holeKapton.name() << " number 1 positioned in " << kapton.name() << " at (0,0," << zpos
                          << ") with no rotation";

      // Wafer
      name = waferName[k];
      if (k == 0 && dlHybrid < dlTop) {
        bl1 = 0.5 * dlTop;
        bl2 = 0.5 * dlBottom;
      } else {
        bl1 = 0.5 * dlBottom;
        bl2 = 0.5 * dlTop;
      }
      h1 = 0.5 * waferThick[k];
      dz = 0.5 * fullHeight;
      solid = ns.addSolidNS(name, Trap(dz, 0, 0, h1, bl1, bl1, 0, h1, bl2, bl2, 0));
      LogDebug("TIDGeom") << solid.name() << " Trap made of " << waferMat << " of dimensions " << dz << ", 0, 0, " << h1
                          << ", " << bl1 << ", " << bl1 << ", 0, " << h1 << ", " << bl2 << ", " << bl2 << ", 0";
      Volume wafer = ns.addVolumeNS(Volume(name, solid, ns.material(waferMat)));

      // Active
      name = activeName[k];
      if (k == 0 && dlHybrid < dlTop) {
        bl1 -= sideWidthTop;
        bl2 -= sideWidthBottom;
      } else {
        bl1 -= sideWidthBottom;
        bl2 -= sideWidthTop;
      }
      dz = 0.5 * (waferThick[k] - backplaneThick[k]);  // inactive backplane
      h1 = 0.5 * activeHeight;
      solid = ns.addSolidNS(name, Trap(dz, 0, 0, h1, bl2, bl1, 0, h1, bl2, bl1, 0));
      LogDebug("TIDGeom") << solid.name() << " Trap made of " << activeMat << " of dimensions " << dz << ", 0, 0, "
                          << h1 << ", " << bl2 << ", " << bl1 << ", 0, " << h1 << ", " << bl2 << ", " << bl1 << ", 0";
      Volume active = ns.addVolumeNS(Volume(name, solid, ns.material(activeMat)));
      rot = ns.rotation(activeRot);
      Position tran(0.0, -0.5 * backplaneThick[k], 0.0);     // from the definition of the wafer local axes
      wafer.placeVolume(active, 1, Transform3D(rot, tran));  // inactive backplane copyNr=1
      LogDebug("TIDGeom") << "DDTIDModuleAlgo test: " << active.name() << " number 1 positioned in " << wafer.name()
                          << " at " << tran << " with " << rot;

      //Pitch Adapter
      name = pitchName[k];
      if (dlHybrid > dlTop) {
        dz = 0.5 * dlTop;
      } else {
        dz = 0.5 * dlBottom;
      }
      if (k == 0) {
        dx = dz;
        dy = 0.5 * pitchThick;
        dz = 0.5 * pitchHeight;
        solid = ns.addSolidNS(name, Box(dx, dy, dz));
        LogDebug("TIDGeom") << solid.name() << " Box made of " << pitchMat << " of dimensions"
                            << " " << dx << ", " << dy << ", " << dz;
      } else {
        h1 = 0.5 * pitchThick;
        bl1 = 0.5 * pitchHeight + 0.5 * dz * sin(detTilt);
        bl2 = 0.5 * pitchHeight - 0.5 * dz * sin(detTilt);

        dz -= 0.5 * pitchStereoTol;
        bl1 -= pitchStereoTol;
        bl2 -= pitchStereoTol;

        double thet = atan((bl1 - bl2) / (2. * dz));
        solid = ns.addSolidNS(name, Trap(dz, thet, 0, h1, bl1, bl1, 0, h1, bl2, bl2, 0));
        LogDebug("TIDGeom") << solid.name() << " Trap made of " << pitchMat << " of "
                            << "dimensions " << dz << ", " << convertRadToDeg(thet) << ", 0, " << h1 << ", " << bl1
                            << ", " << bl1 << ", 0, " << h1 << ", " << bl2 << ", " << bl2 << ", 0";
      }
      /* Volume pa = */ ns.addVolumeNS(Volume(name, solid, ns.material(pitchMat)));
    }
  }
  LogDebug("TIDGeom") << "<<== End of DDTIDModuleAlgo construction ...";
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTIDModuleAlgo, algorithm)
