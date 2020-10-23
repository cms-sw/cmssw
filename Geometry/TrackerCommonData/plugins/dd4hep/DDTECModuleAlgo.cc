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
  Volume mother = ns.volume(args.parentName());

  //variables:
  //double         noOverlapShift   = args.value<double>("NoOverlapShift");
  int ringNo = args.value<int>("RingNo");
  bool isStereo = args.value<int>("isStereo") == 1;
  bool isRing6 = (ringNo == 6);
  double rPos =
      args.value<double>("RPos");  //Position in R relativ to the center of the TEC ( this is the coord-sys of Tubs)
  double posCorrectionPhi = isStereo ? args.value<double>("PosCorrectionPhi")
                                     : 0e0;  // the Phi position of the stereo Modules has to be corrected
  string standardRot = args.value<string>(
      "StandardRotation");  //Rotation that aligns the mother(Tub ) coordinate System with the components
  string genMat = args.value<string>("GeneralMaterial");                              //General material name
  double moduleThick = args.value<double>("ModuleThick");                             //Module thickness
  double detTilt = args.value<double>("DetTilt");                                     //Tilt of stereo detector
  double fullHeight = args.value<double>("FullHeight");                               //Height
  double dlTop = args.value<double>("DlTop");                                         //Width at top of wafer
  double dlBottom = args.value<double>("DlBottom");                                   //Width at bottom of wafer
  double dlHybrid = args.value<double>("DlHybrid");                                   //Width at the hybrid end
  double frameWidth = args.value<double>("FrameWidth");                               //Frame         width
  double frameThick = args.value<double>("FrameThick");                               //              thickness
  double frameOver = args.value<double>("FrameOver");                                 //              overlap (on sides)
  string topFrameMat = args.value<string>("TopFrameMaterial");                        //Top frame     material
  double topFrameHeight = args.value<double>("TopFrameHeight");                       //              height
  double topFrameThick = args.value<double>("TopFrameThick");                         //              thickness
  double topFrameTopWidth = args.value<double>("TopFrameTopWidth");                   //             Width at the top
  double topFrameBotWidth = args.value<double>("TopFrameBotWidth");                   //             Width at the bottom
  double topFrame2Width = isStereo ? args.value<double>("TopFrame2Width") : 0e0;      //  Stereo:2ndPart   Width
  double topFrame2LHeight = isStereo ? args.value<double>("TopFrame2LHeight") : 0e0;  //             left  height
  double topFrame2RHeight = isStereo ? args.value<double>("TopFrame2RHeight") : 0e0;  //             right height
  double topFrameZ = args.value<double>("TopFrameZ");                                 //              z-positions

  double resizeH = 0.96;
  string sideFrameMat = args.value<string>("SideFrameMaterial");   //Side frame    material
  double sideFrameThick = args.value<double>("SideFrameThick");    //              thickness
  double sideFrameLWidth = args.value<double>("SideFrameLWidth");  //    Left     Width (for stereo modules upper one)
  double sideFrameLWidthLow = isStereo ? args.value<double>("SideFrameLWidthLow")
                                       : 0e0;  //           Width (only for stereo modules: lower Width)
  double sideFrameLHeight = resizeH * args.value<double>("SideFrameLHeight");  //             Height
  double sideFrameLtheta = args.value<double>("SideFrameLtheta");  //              angle of the trapezoid shift
  double sideFrameRWidth = args.value<double>("SideFrameRWidth");  //    Right    Width (for stereo modules upper one)
  double sideFrameRWidthLow = isStereo ? args.value<double>("SideFrameRWidthLow")
                                       : 0e0;  //           Width (only for stereo modules: lower Width)
  double sideFrameRHeight = resizeH * args.value<double>("SideFrameRHeight");  //             Height
  double sideFrameRtheta = args.value<double>("SideFrameRtheta");  //              angle of the trapezoid shift
  vector<double> siFrSuppBoxWidth = args.value<vector<double> >("SiFrSuppBoxWidth");    //    Supp.Box Width
  vector<double> siFrSuppBoxHeight = args.value<vector<double> >("SiFrSuppBoxHeight");  //            Height
  vector<double> siFrSuppBoxYPos = args.value<vector<double> >(
      "SiFrSuppBoxYPos");  //              y-position of the supplies box (with HV an thermal sensor...)
  double sideFrameZ = args.value<double>("SideFrameZ");               //              z-positions
  double siFrSuppBoxThick = args.value<double>("SiFrSuppBoxThick");   //             thickness
  string siFrSuppBoxMat = args.value<string>("SiFrSuppBoxMaterial");  //              material
  string waferMat = args.value<string>("WaferMaterial");              //Wafer         material
  double waferPosition = args.value<double>(
      "WaferPosition");  //              position of the wafer (was formaly done by adjusting topFrameHeigt)
  double sideWidthTop = args.value<double>("SideWidthTop");        //              widths on the side Top
  double sideWidthBottom = args.value<double>("SideWidthBottom");  //                                 Bottom
  string waferRot = args.value<string>("WaferRotation");           //              rotation matrix
  string activeMat = args.value<string>("ActiveMaterial");         //Sensitive     material
  double activeHeight = args.value<double>("ActiveHeight");        //              height
  double waferThick = args.value<double>("WaferThick");     //              wafer thickness (active = wafer - backplane)
  string activeRot = args.value<string>("ActiveRotation");  //              Rotation matrix
  double activeZ = args.value<double>("ActiveZ");           //              z-positions
  double backplaneThick = args.value<double>("BackPlaneThick");               //              thickness
  double inactiveDy = ringNo > 3 ? args.value<double>("InactiveDy") : 0e0;    //InactiveStrip  Hight of ( rings > 3)
  double inactivePos = ringNo > 3 ? args.value<double>("InactivePos") : 0e0;  //               y-Position
  string inactiveMat = ringNo > 3 ? args.value<string>("InactiveMaterial") : string();  //               material
  string hybridMat = args.value<string>("HybridMaterial");                              //Hybrid        material
  double hybridHeight = args.value<double>("HybridHeight");                             //              height
  double hybridWidth = args.value<double>("HybridWidth");                               //              width
  double hybridThick = args.value<double>("HybridThick");                               //              thickness
  double hybridZ = args.value<double>("HybridZ");                                       //              z-positions
  string pitchMat = args.value<string>("PitchMaterial");                                //Pitch adapter material
  double pitchWidth = args.value<double>("PitchWidth");                                 //              width
  double pitchHeight = args.value<double>("PitchHeight");                               //              height
  double pitchThick = args.value<double>("PitchThick");                                 //              thickness
  double pitchZ = args.value<double>("PitchZ");                                         //              z-positions
  string pitchRot = args.value<string>("PitchRotation");                                //              rotation matrix
  string bridgeMat = args.value<string>("BridgeMaterial");                              //Bridge        material
  double bridgeWidth = args.value<double>("BridgeWidth");                               //              width
  double bridgeThick = args.value<double>("BridgeThick");                               //              thickness
  double bridgeHeight = args.value<double>("BridgeHeight");                             //              height
  double bridgeSep = args.value<double>("BridgeSeparation");                            //              separation
  vector<double> siReenforceHeight = args.value<vector<double> >("SiReenforcementHeight");  // SiReenforcement Height
  vector<double> siReenforceWidth = args.value<vector<double> >("SiReenforcementWidth");    //             Width
  vector<double> siReenforceYPos = args.value<vector<double> >("SiReenforcementPosY");      //              Y - Position
  double siReenforceThick = args.value<double>("SiReenforcementThick");                     //             Thick
  string siReenforceMat = args.value<string>("SiReenforcementMaterial");                    //             Materieal

  edm::LogVerbatim("TECGeom") << "debug: ModuleThick " << moduleThick << " Detector Tilt " << convertRadToDeg(detTilt)
                              << " Height " << fullHeight << " dl(Top) " << dlTop << " dl(Bottom) " << dlBottom
                              << " dl(Hybrid) " << dlHybrid << " rPos " << rPos << " standrad rotation " << standardRot;
  edm::LogVerbatim("TECGeom") << "debug: Frame Width " << frameWidth << " Thickness " << frameThick << " Overlap "
                              << frameOver;
  edm::LogVerbatim("TECGeom") << "debug: Top Frame Material " << topFrameMat << " Height " << topFrameHeight
                              << " Top Width " << topFrameTopWidth << " Bottom Width " << topFrameTopWidth
                              << " Thickness " << topFrameThick << " positioned at" << topFrameZ;
  edm::LogVerbatim("TECGeom") << "debug : Side Frame Material " << sideFrameMat << " Thickness " << sideFrameThick
                              << " left Leg's Width: " << sideFrameLWidth << " left Leg's Height: " << sideFrameLHeight
                              << " left Leg's tilt(theta): " << sideFrameLtheta
                              << " right Leg's Width: " << sideFrameRWidth
                              << " right Leg's Height: " << sideFrameRHeight
                              << " right Leg's tilt(theta): " << sideFrameRtheta
                              << "Supplies Box's Material: " << siFrSuppBoxMat << " positioned at" << sideFrameZ;
  for (int i = 0; i < (int)(siFrSuppBoxWidth.size()); i++)
    edm::LogVerbatim("TECGeom") << " Supplies Box" << i << "'s Width: " << siFrSuppBoxWidth[i] << " Supplies Box" << i
                                << "'s Height: " << siFrSuppBoxHeight[i] << " Supplies Box" << i
                                << "'s y Position: " << siFrSuppBoxYPos[i];
  edm::LogVerbatim("TECGeom") << "debug: Wafer Material " << waferMat << " Side Width Top" << sideWidthTop
                              << " Side Width Bottom" << sideWidthBottom << " and positioned at " << waferPosition
                              << " positioned with rotation"
                              << " matrix:" << waferRot;
  edm::LogVerbatim("TECGeom") << "debug: Active Material " << activeMat << " Height " << activeHeight << " rotated by "
                              << activeRot << " translated by (0,0," << -0.5 * backplaneThick << ")"
                              << " Thickness/Z" << waferThick - backplaneThick << "/" << activeZ;
  edm::LogVerbatim("TECGeom") << "debug: Hybrid Material " << hybridMat << " Height " << hybridHeight << " Width "
                              << hybridWidth << " Thickness " << hybridThick << " Z" << hybridZ;
  edm::LogVerbatim("TECGeom") << "debug: Pitch Adapter Material " << pitchMat << " Height " << pitchHeight
                              << " Thickness " << pitchThick << " position with "
                              << " rotation " << pitchRot << " at Z" << pitchZ;
  edm::LogVerbatim("TECGeom") << "debug: Bridge Material " << bridgeMat << " Width " << bridgeWidth << " Thickness "
                              << bridgeThick << " Height " << bridgeHeight << " Separation " << bridgeSep;
  edm::LogVerbatim("TECGeom") << "FALTBOOT DDTECModuleAlgo debug : Si-Reenforcement Material " << sideFrameMat
                              << " Thickness " << siReenforceThick;
  for (int i = 0; i < (int)(siReenforceWidth.size()); i++)
    edm::LogVerbatim("TECGeom") << " SiReenforcement" << i << "'s Width: " << siReenforceWidth[i] << " SiReenforcement"
                                << i << "'s Height: " << siReenforceHeight[i] << " SiReenforcement" << i
                                << "'s y Position: " << siReenforceYPos[i];

  if (!isStereo) {
    edm::LogVerbatim("TECGeom") << "This is a normal module, in ring " << ringNo << "!";
  } else {
    edm::LogVerbatim("TECGeom") << "This is a stereo module, in ring " << ringNo << "!";
    edm::LogVerbatim("TECGeom") << "Phi Position corrected by " << posCorrectionPhi << "*rad";
    edm::LogVerbatim("TECGeom") << "debug: stereo Top Frame 2nd Part left Heigt " << topFrame2LHeight
                                << " right Height " << topFrame2RHeight << " Width " << topFrame2Width;
    edm::LogVerbatim("TECGeom") << " left Leg's lower Width: " << sideFrameLWidthLow
                                << " right Leg's lower Width: " << sideFrameRWidthLow;
  }

  // Execution part:

  edm::LogVerbatim("TECGeom") << "==>> Constructing DDTECModuleAlgo: ";
  //declarations
  double tmp;
  //names
  string name;
  string tag("Rphi");
  if (isStereo)
    tag = "Stereo";
  //usefull constants
  const double topFrameEndZ = 0.5 * (-waferPosition + fullHeight) + pitchHeight + hybridHeight - topFrameHeight;
  string idName = ns.prepend(ns.realName(mother.name()));
  edm::LogVerbatim("TECGeom") << "idName: " << idName << " parent " << mother.name() << " namespace " << ns.name();
  Solid solid;

  //set global parameters
  Material matter = ns.material(genMat);
  double dzdif = fullHeight + topFrameHeight;
  if (isStereo)
    dzdif += 0.5 * (topFrame2LHeight + topFrame2RHeight);

  double dxbot = 0.5 * dlBottom + frameWidth - frameOver;
  double dxtop = 0.5 * dlHybrid + frameWidth - frameOver;
  //  topfr = 0.5*dlBottom * sin(detTilt);
  if (isRing6) {
    dxbot = dxtop;
    dxtop = 0.5 * dlTop + frameWidth - frameOver;
    //    topfr = 0.5*dlTop    * sin(detTilt);
  }
  double dxdif = dxtop - dxbot;

  //Frame Sides
  // left Frame
  name = idName + "SideFrameLeft";
  double h1 = 0.5 * sideFrameThick;
  double dz = 0.5 * sideFrameLHeight;
  double bl1 = 0.5 * sideFrameLWidth;
  double bl2 = bl1;
  double thet = sideFrameLtheta;
  //for stereo modules
  if (isStereo)
    bl1 = 0.5 * sideFrameLWidthLow;
  solid = Trap(dz, thet, 0, h1, bl1, bl1, 0, h1, bl2, bl2, 0);
  ns.addSolidNS(name, solid);
  edm::LogVerbatim("TECGeom") << "Solid: " << name << " " << solid.name() << " Trap made of " << sideFrameMat
                              << " of dimensions " << dz << ",  " << thet << ", 0, " << h1 << ", " << bl1 << ", " << bl1
                              << ", 0, " << h1 << ", " << bl2 << ", " << bl2 << ", 0";
  Volume sideFrameLeft(name, solid, ns.material(sideFrameMat));
  ns.addVolumeNS(sideFrameLeft);
  //translate
  double xpos = -0.5 * topFrameBotWidth + bl2 + tan(fabs(thet)) * dz;
  double ypos = sideFrameZ;
  double zpos = topFrameEndZ - dz;
  //flip ring 6
  if (isRing6) {
    zpos *= -1;
    xpos -= 2 * tan(fabs(thet)) * dz;  // because of the flip the tan(..) to be in the other direction
  }
  //the stereo modules are on the back of the normal ones...
  if (isStereo) {
    xpos = -0.5 * topFrameBotWidth + bl2 * cos(detTilt) + dz * sin(fabs(thet) + detTilt) / cos(fabs(thet));
    xpos = -xpos;
    zpos = topFrameEndZ - topFrame2LHeight - 0.5 * sin(detTilt) * (topFrameBotWidth - topFrame2Width) -
           dz * cos(detTilt + fabs(thet)) / cos(fabs(thet)) + bl2 * sin(detTilt) - 0.1_mm;
  }
  //position
  mother.placeVolume(
      sideFrameLeft,
      isStereo ? 2 : 1,
      dd4hep::Transform3D(ns.rotation(waferRot),
                          dd4hep::Position(zpos + rPos, isStereo ? xpos + rPos * sin(posCorrectionPhi) : xpos, ypos)));

  //right Frame
  name = idName + "SideFrameRight";
  h1 = 0.5 * sideFrameThick;
  dz = 0.5 * sideFrameRHeight;
  bl1 = bl2 = 0.5 * sideFrameRWidth;
  thet = sideFrameRtheta;
  if (isStereo)
    bl1 = 0.5 * sideFrameRWidthLow;
  solid = Trap(dz, thet, 0, h1, bl1, bl1, 0, h1, bl2, bl2, 0);
  ns.addSolidNS(name, solid);
  edm::LogVerbatim("TECGeom") << "Solid:\t" << name << " " << solid.name() << " Trap made of " << sideFrameMat
                              << " of dimensions " << dz << ", " << thet << ", 0, " << h1 << ", " << bl1 << ", " << bl1
                              << ", 0, " << h1 << ", " << bl2 << ", " << bl2 << ", 0";
  Volume sideFrameRight(name, solid, ns.material(sideFrameMat));
  ns.addVolumeNS(sideFrameRight);
  //translate
  xpos = 0.5 * topFrameBotWidth - bl2 - tan(fabs(thet)) * dz;
  ypos = sideFrameZ;
  zpos = topFrameEndZ - dz;
  if (isRing6) {
    zpos *= -1;
    xpos += 2 * tan(fabs(thet)) * dz;  // because of the flip the tan(..) has to be in the other direction
  }
  if (isStereo) {
    xpos = 0.5 * topFrameBotWidth - bl2 * cos(detTilt) - dz * sin(fabs(detTilt - fabs(thet))) / cos(fabs(thet));
    xpos = -xpos;
    zpos = topFrameEndZ - topFrame2RHeight + 0.5 * sin(detTilt) * (topFrameBotWidth - topFrame2Width) -
           dz * cos(detTilt - fabs(thet)) / cos(fabs(thet)) - bl2 * sin(detTilt) - 0.1_mm;
  }
  //position it
  mother.placeVolume(
      sideFrameRight,
      isStereo ? 2 : 1,
      dd4hep::Transform3D(ns.rotation(waferRot),
                          dd4hep::Position(zpos + rPos, isStereo ? xpos + rPos * sin(posCorrectionPhi) : xpos, ypos)));
  //Supplies Box(es)
  matter = ns.material(siFrSuppBoxMat);
  for (int i = 0; i < (int)(siFrSuppBoxWidth.size()); i++) {
    name = idName + "SuppliesBox" + std::to_string(i);

    h1 = 0.5 * siFrSuppBoxThick;
    dz = 0.5 * siFrSuppBoxHeight[i];
    bl1 = bl2 = 0.5 * siFrSuppBoxWidth[i];
    thet = sideFrameRtheta;
    if (isStereo)
      thet = -atan(fabs(sideFrameRWidthLow - sideFrameRWidth) / (2 * sideFrameRHeight) - tan(fabs(thet)));
    // ^-- this calculates the lower left angel of the tipped trapezoid, which is the SideFframe...

    solid = Trap(dz, thet, 0, h1, bl1, bl1, 0, h1, bl2, bl2, 0);
    edm::LogVerbatim("TECGeom") << "Solid:\t" << name << " " << solid.name() << " Trap made of " << siFrSuppBoxMat
                                << " of dimensions " << dz << ", 0, 0, " << h1 << ", " << bl1 << ", " << bl1 << ", 0, "
                                << h1 << ", " << bl2 << ", " << bl2 << ", 0";
    Volume siFrSuppBox(name, solid, matter);
    ns.addVolumeNS(siFrSuppBox);
    //translate
    xpos = 0.5 * topFrameBotWidth - sideFrameRWidth - bl1 - siFrSuppBoxYPos[i] * tan(fabs(thet));
    ypos = sideFrameZ *
           (0.5 + (siFrSuppBoxThick / sideFrameThick));  //via * so I do not have to worry about the sign of sideFrameZ
    zpos = topFrameEndZ - siFrSuppBoxYPos[i];
    if (isRing6) {
      xpos += 2 * fabs(tan(thet)) * siFrSuppBoxYPos[i];  // the flipped issue again
      zpos *= -1;
    }
    if (isStereo) {
      xpos = 0.5 * topFrameBotWidth - (sideFrameRWidth + bl1) * cos(detTilt) -
             sin(fabs(detTilt - fabs(thet))) *
                 (siFrSuppBoxYPos[i] + dz * (1 / cos(thet) - cos(detTilt)) + bl1 * sin(detTilt));
      xpos = -xpos;
      zpos = topFrameEndZ - topFrame2RHeight - 0.5 * sin(detTilt) * (topFrameBotWidth - topFrame2Width) -
             siFrSuppBoxYPos[i] - sin(detTilt) * sideFrameRWidth;
    }
    //position it;
    mother.placeVolume(siFrSuppBox,
                       isStereo ? 2 : 1,
                       dd4hep::Transform3D(
                           ns.rotation(waferRot),
                           dd4hep::Position(zpos + rPos, isStereo ? xpos + rPos * sin(posCorrectionPhi) : xpos, ypos)));
  }

  //The Hybrid
  name = idName + "Hybrid";
  double dx = 0.5 * hybridWidth;
  double dy = 0.5 * hybridThick;
  dz = 0.5 * hybridHeight;
  solid = Box(dx, dy, dz);
  ns.addSolidNS(name, solid);
  edm::LogVerbatim("TECGeom") << "Solid:\t" << name << " " << solid.name() << " Box made of " << hybridMat
                              << " of dimensions " << dx << ", " << dy << ", " << dz;
  Volume hybrid(name, solid, ns.material(hybridMat));
  ns.addVolumeNS(hybrid);

  ypos = hybridZ;
  zpos = 0.5 * (-waferPosition + fullHeight + hybridHeight) + pitchHeight;
  if (isRing6)
    zpos *= -1;
  //position it
  mother.placeVolume(
      hybrid,
      isStereo ? 2 : 1,
      dd4hep::Transform3D(ns.rotation(standardRot),
                          dd4hep::Position(zpos + rPos, isStereo ? rPos * sin(posCorrectionPhi) : 0., ypos)));

  // Wafer
  name = idName + tag + "Wafer";
  bl1 = 0.5 * dlBottom;
  bl2 = 0.5 * dlTop;
  h1 = 0.5 * waferThick;
  dz = 0.5 * fullHeight;
  solid = Trap(dz, 0, 0, h1, bl1, bl1, 0, h1, bl2, bl2, 0);
  ns.addSolidNS(name, solid);
  edm::LogVerbatim("TECGeom") << "Solid:\t" << name << " " << solid.name() << " Trap made of " << waferMat
                              << " of dimensions " << dz << ", 0, 0, " << h1 << ", " << bl1 << ", " << bl1 << ", 0, "
                              << h1 << ", " << bl2 << ", " << bl2 << ", 0";
  Volume wafer(name, solid, ns.material(waferMat));

  ypos = activeZ;
  zpos = -0.5 * waferPosition;  // former and incorrect topFrameHeight;
  if (isRing6)
    zpos *= -1;

  mother.placeVolume(
      wafer,
      isStereo ? 2 : 1,
      dd4hep::Transform3D(ns.rotation(waferRot),
                          dd4hep::Position(zpos + rPos, isStereo ? rPos * sin(posCorrectionPhi) : 0., ypos)));

  // Active
  name = idName + tag + "Active";
  bl1 -= sideWidthBottom;
  bl2 -= sideWidthTop;
  dz = 0.5 * (waferThick - backplaneThick);  // inactive backplane
  h1 = 0.5 * activeHeight;
  if (isRing6) {  //switch bl1 <->bl2
    tmp = bl2;
    bl2 = bl1;
    bl1 = tmp;
  }
  solid = Trap(dz, 0, 0, h1, bl2, bl1, 0, h1, bl2, bl1, 0);
  ns.addSolidNS(name, solid);
  edm::LogVerbatim("TECGeom") << "Solid:\t" << name << " " << solid.name() << " Trap made of " << activeMat
                              << " of dimensions " << dz << ", 0, 0, " << h1 << ", " << bl2 << ", " << bl1 << ", 0, "
                              << h1 << ", " << bl2 << ", " << bl1 << ", 0";
  Volume active(name, solid, ns.material(activeMat));
  ns.addVolumeNS(active);

  wafer.placeVolume(
      active, 1, dd4hep::Transform3D(ns.rotation(activeRot), dd4hep::Position(0., -0.5 * backplaneThick, 0.)));

  //inactive part in rings > 3
  if (ringNo > 3) {
    inactivePos -= fullHeight - activeHeight;  //inactivePos is measured from the beginning of the _wafer_
    name = idName + tag + "Inactive";
    bl1 = 0.5 * dlBottom - sideWidthBottom +
          ((0.5 * dlTop - sideWidthTop - 0.5 * dlBottom + sideWidthBottom) / activeHeight) *
              (activeHeight - inactivePos - inactiveDy);
    bl2 = 0.5 * dlBottom - sideWidthBottom +
          ((0.5 * dlTop - sideWidthTop - 0.5 * dlBottom + sideWidthBottom) / activeHeight) *
              (activeHeight - inactivePos + inactiveDy);
    dz = 0.5 * (waferThick - backplaneThick);  // inactive backplane
    h1 = inactiveDy;
    if (isRing6) {  //switch bl1 <->bl2
      tmp = bl2;
      bl2 = bl1;
      bl1 = tmp;
    }
    solid = Trap(dz, 0, 0, h1, bl2, bl1, 0, h1, bl2, bl1, 0);
    ns.addSolidNS(name, solid);
    edm::LogVerbatim("TECGeom") << "Solid:\t" << name << " " << solid.name() << " Trap made of " << inactiveMat
                                << " of dimensions " << dz << ", 0, 0, " << h1 << ", " << bl2 << ", " << bl1 << ", 0, "
                                << h1 << ", " << bl2 << ", " << bl1 << ", 0";
    Volume inactive(name, solid, ns.material(inactiveMat));
    ns.addVolumeNS(inactive);
    ypos = inactivePos - 0.5 * activeHeight;

    active.placeVolume(inactive, 1, dd4hep::Position(0., ypos, 0.));
  }
  //Pitch Adapter
  name = idName + "PA";
  if (!isStereo) {
    dx = 0.5 * pitchWidth;
    dy = 0.5 * pitchThick;
    dz = 0.5 * pitchHeight;
    solid = Box(dx, dy, dz);
    ns.addSolidNS(name, solid);
    edm::LogVerbatim("TECGeom") << "Solid:\t" << name << " " << solid.name() << " Box made of " << pitchMat
                                << " of dimensions " << dx << ", " << dy << ", " << dz;
  } else {
    dz = 0.5 * pitchWidth;
    h1 = 0.5 * pitchThick;
    bl1 = 0.5 * pitchHeight + 0.5 * dz * sin(detTilt);
    bl2 = 0.5 * pitchHeight - 0.5 * dz * sin(detTilt);
    thet = atan((bl1 - bl2) / (2. * dz));
    solid = Trap(dz, thet, 0, h1, bl1, bl1, 0, h1, bl2, bl2, 0);
    ns.addSolidNS(name, solid);
    edm::LogVerbatim("TECGeom") << "Solid:\t" << name << " " << solid.name() << " Trap made of " << pitchMat
                                << " of dimensions " << dz << ", " << convertRadToDeg(thet) << ", 0, " << h1 << ", "
                                << bl1 << ", " << bl1 << ", 0, " << h1 << ", " << bl2 << ", " << bl2 << ", 0";
  }
  xpos = 0;
  ypos = pitchZ;
  zpos = 0.5 * (-waferPosition + fullHeight + pitchHeight);
  if (isRing6)
    zpos *= -1;
  if (isStereo)
    xpos = 0.5 * fullHeight * sin(detTilt);

  Volume pa(name, solid, ns.material(pitchMat));
  if (isStereo)
    mother.placeVolume(pa,
                       2,
                       dd4hep::Transform3D(ns.rotation(pitchRot),
                                           dd4hep::Position(zpos + rPos, xpos + rPos * sin(posCorrectionPhi), ypos)));
  else
    mother.placeVolume(pa, 1, dd4hep::Transform3D(ns.rotation(standardRot), dd4hep::Position(zpos + rPos, xpos, ypos)));

  //Top of the frame
  name = idName + "TopFrame";
  h1 = 0.5 * topFrameThick;
  dz = 0.5 * topFrameHeight;
  bl1 = 0.5 * topFrameBotWidth;
  bl2 = 0.5 * topFrameTopWidth;
  if (isRing6) {  // ring 6 faces the other way!
    bl1 = 0.5 * topFrameTopWidth;
    bl2 = 0.5 * topFrameBotWidth;
  }

  solid = Trap(dz, 0, 0, h1, bl1, bl1, 0, h1, bl2, bl2, 0);
  ns.addSolid(name, solid);
  edm::LogVerbatim("TECGeom") << "Solid:\t" << name << " " << solid.name() << " Trap made of " << topFrameMat
                              << " of dimensions " << dz << ", 0, 0, " << h1 << ", " << bl1 << ", " << bl1 << ", 0, "
                              << h1 << ", " << bl2 << ", " << bl2 << ", 0";
  Volume topFrame(name, solid, ns.material(topFrameMat));
  ns.addVolumeNS(topFrame);

  if (isStereo) {
    name = idName + "TopFrame2";
    //additional object to build the not trapzoid geometry of the stereo topframes
    dz = 0.5 * topFrame2Width;
    h1 = 0.5 * topFrameThick;
    bl1 = 0.5 * topFrame2LHeight;
    bl2 = 0.5 * topFrame2RHeight;
    thet = atan((bl1 - bl2) / (2. * dz));

    solid = Trap(dz, thet, 0, h1, bl1, bl1, 0, h1, bl2, bl2, 0);
    ns.addSolid(name, solid);
    edm::LogVerbatim("TECGeom") << "Solid:\t" << name << " " << solid.name() << " Trap made of " << topFrameMat
                                << " of dimensions " << dz << ", " << convertRadToDeg(thet) << ", 0, " << h1 << ", "
                                << bl1 << ", " << bl1 << ", 0, " << h1 << ", " << bl2 << ", " << bl2 << ", 0";
  }

  // Position the topframe
  ypos = topFrameZ;
  zpos = 0.5 * (-waferPosition + fullHeight - topFrameHeight) + pitchHeight + hybridHeight;
  if (isRing6) {
    zpos *= -1;
  }

  mother.placeVolume(
      topFrame,
      isStereo ? 2 : 1,
      dd4hep::Transform3D(ns.rotation(standardRot),
                          dd4hep::Position(zpos + rPos, isStereo ? rPos * sin(posCorrectionPhi) : 0., ypos)));
  if (isStereo) {
    //create
    Volume topFrame2(name, solid, ns.material(topFrameMat));
    zpos -= 0.5 * (topFrameHeight + 0.5 * (topFrame2LHeight + topFrame2RHeight));
    mother.placeVolume(
        topFrame2,
        2,
        dd4hep::Transform3D(ns.rotation(pitchRot), dd4hep::Position(zpos + rPos, rPos * sin(posCorrectionPhi), ypos)));
  }

  //Si - Reencorcement
  matter = ns.material(siReenforceMat);
  for (int i = 0; i < (int)(siReenforceWidth.size()); i++) {
    name = idName + "SiReenforce" + std::to_string(i);
    h1 = 0.5 * siReenforceThick;
    dz = 0.5 * siReenforceHeight[i];
    bl1 = bl2 = 0.5 * siReenforceWidth[i];
    solid = Trap(dz, 0, 0, h1, bl1, bl1, 0, h1, bl2, bl2, 0);
    edm::LogVerbatim("TECGeom") << "Solid:\t" << name << " " << solid.name() << " Trap made of " << matter.name()
                                << " of dimensions " << dz << ", 0, 0, " << h1 << ", " << bl1 << ", " << bl1 << ", 0, "
                                << h1 << ", " << bl2 << ", " << bl2 << ", 0";
    Volume siReenforce(name, solid, matter);
    ns.addVolumeNS(siReenforce);
    //translate
    xpos = 0;
    ypos = sideFrameZ;
    zpos = topFrameEndZ - dz - siReenforceYPos[i];

    if (isRing6)
      zpos *= -1;
    if (isStereo) {
      xpos = (-siReenforceYPos[i] + 0.5 * fullHeight) * sin(detTilt);
      //  thet = detTilt;
      //  if(topFrame2RHeight > topFrame2LHeight) thet *= -1;
      //    zpos -= topFrame2RHeight + sin(thet)*(sideFrameRWidth + 0.5*dlTop);
      zpos -= topFrame2RHeight + sin(fabs(detTilt)) * 0.5 * topFrame2Width;
    }

    mother.placeVolume(siReenforce,
                       isStereo ? 2 : 1,
                       dd4hep::Transform3D(
                           ns.rotation(waferRot),
                           dd4hep::Position(zpos + rPos, isStereo ? xpos + rPos * sin(posCorrectionPhi) : xpos, ypos)));
  }

  //Bridge
  if (bridgeMat != "None") {
    name = idName + "Bridge";
    bl2 = 0.5 * bridgeSep + bridgeWidth;
    bl1 = bl2 - bridgeHeight * dxdif / dzdif;
    h1 = 0.5 * bridgeThick;
    dz = 0.5 * bridgeHeight;
    solid = Trap(dz, 0, 0, h1, bl1, bl1, 0, h1, bl2, bl2, 0);
    edm::LogVerbatim("TECGeom") << "Solid:\t" << name << " " << solid.name() << " Trap made of " << bridgeMat
                                << " of dimensions " << dz << ", 0, 0, " << h1 << ", " << bl1 << ", " << bl1 << ", 0, "
                                << h1 << ", " << bl2 << ", " << bl2 << ", 0";
    Volume bridge(name, solid, ns.material(bridgeMat));
    ns.addVolumeNS(bridge);

    name = idName + "BridgeGap";
    bl1 = 0.5 * bridgeSep;
    solid = Box(bl1, h1, dz);
    edm::LogVerbatim("TECGeom") << "Solid:\t" << name << " " << solid.name() << " Box made of " << genMat
                                << " of dimensions " << bl1 << ", " << h1 << ", " << dz;
    Volume bridgeGap(name, solid, ns.material(genMat));
    ns.addVolumeNS(bridgeGap);
    /* PlacedVolume pv = */ bridge.placeVolume(bridgeGap, 1);
    edm::LogVerbatim("TECGeom") << "Solid: " << bridgeGap.name() << " number 1 positioned in " << bridge.name()
                                << " at (0,0,0) with no rotation";
  }
  edm::LogVerbatim("TECGeom") << "<<== End of DDTECModuleAlgo construction ...";
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTECModuleAlgo, algorithm)
