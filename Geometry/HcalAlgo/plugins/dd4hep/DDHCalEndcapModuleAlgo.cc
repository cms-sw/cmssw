///////////////////////////////////////////////////////////////////////////////
// File: DDHCalEndcapModuleAlgo.cc
//   adapted from CCal(G4)HcalEndcap.cc
// Description: Geometry factory class for Hcal Endcap
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

#include "DataFormats/Math/interface/GeantUnits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

struct HCalEndcapModuleAlgo {
  std::string genMaterial;             //General material
  std::string absorberMat;             //Absorber material
  std::string plasticMat;              //Plastic material cover
  std::string scintMat;                //Scintillator material
  std::string rotstr;                  //Rotation matrix to place in mother
  int sectors;                         //Number of potenital straight edges
  double zMinBlock;                    //Minimum z extent of the block
  double zMaxBlock;                    //Maximum
  double z1Beam;                       //Position of gap end   along z-axis
  double ziDip;                        //Starting Z of dipped part of body
  double dzStep;                       //Width in Z of a layer
  double moduleThick;                  //Thickness of a layer (air/absorber)
  double layerThick;                   //Thickness of a layer (plastic)
  double scintThick;                   //Thickness of scinitllator
  double rMaxBack;                     //Maximum R after  the dip
  double rMaxFront;                    //Maximum R before the dip
  double slopeBot;                     //Slope of the bottom edge
  double slopeTop;                     //Slope of the top edge
  double slopeTopF;                    //Slope of the top front edge
  double trimLeft;                     //Trim of the left edge
  double trimRight;                    //Trim of the right edge
  double tolAbs;                       //Tolerance for absorber
  int modType;                         //Type of module
  int modNumber;                       //Module number
  int layerType;                       //Layer type
  std::vector<int> layerNumber;        //layer numbers
  std::vector<std::string> phiName;    //Name of Phi sections
  std::vector<std::string> layerName;  //Layer Names

  std::string idName;   //Name of the "parent" volume.
  std::string modName;  //Module Name
  int idOffset;         // Geant4 ID's...    = 4000;

  struct HcalEndcapPar {
    double yh1, bl1, tl1, yh2, bl2, tl2, alp, theta, phi, xpos, ypos, zpos;
    HcalEndcapPar(double yh1v = 0,
                  double bl1v = 0,
                  double tl1v = 0,
                  double yh2v = 0,
                  double bl2v = 0,
                  double tl2v = 0,
                  double alpv = 0,
                  double thv = 0,
                  double fiv = 0,
                  double x = 0,
                  double y = 0,
                  double z = 0)
        : yh1(yh1v),
          bl1(bl1v),
          tl1(tl1v),
          yh2(yh2v),
          bl2(bl2v),
          tl2(tl2v),
          alp(alpv),
          theta(thv),
          phi(fiv),
          xpos(x),
          ypos(y),
          zpos(z) {}
  };

  HCalEndcapModuleAlgo() = delete;

  HCalEndcapModuleAlgo(cms::DDParsingContext& ctxt, xml_h e) {
    cms::DDNamespace ns(ctxt, e, true);
    cms::DDAlgoArguments args(ctxt, e);

    genMaterial = args.value<std::string>("MaterialName");
    absorberMat = args.value<std::string>("AbsorberMat");
    plasticMat = args.value<std::string>("PlasticMat");
    scintMat = args.value<std::string>("ScintMat");
    rotstr = args.value<std::string>("Rotation");
    sectors = args.value<int>("Sectors");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: General material " << genMaterial << "\tAbsorber "
                                 << absorberMat << "\tPlastic " << plasticMat << "\tScintillator " << scintMat
                                 << "\tRotation " << rotstr << "\tSectors " << sectors;
#endif
    zMinBlock = args.value<double>("ZMinBlock");
    zMaxBlock = args.value<double>("ZMaxBlock");
    z1Beam = args.value<double>("Z1Beam");
    ziDip = args.value<double>("ZiDip");
    dzStep = args.value<double>("DzStep");
    moduleThick = args.value<double>("ModuleThick");
    layerThick = args.value<double>("LayerThick");
    scintThick = args.value<double>("ScintThick");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: Zmin " << convertCmToMm(zMinBlock) << "\tZmax "
                                 << convertCmToMm(zMaxBlock) << "\tZ1Beam " << convertCmToMm(z1Beam) << "\tZiDip "
                                 << convertCmToMm(ziDip) << "\tDzStep " << convertCmToMm(dzStep) << "\tModuleThick "
                                 << convertCmToMm(moduleThick) << "\tLayerThick " << convertCmToMm(layerThick)
                                 << "\tScintThick " << convertCmToMm(scintThick);
#endif
    rMaxFront = args.value<double>("RMaxFront");
    rMaxBack = args.value<double>("RMaxBack");
    trimLeft = args.value<double>("TrimLeft");
    trimRight = args.value<double>("TrimRight");
    tolAbs = args.value<double>("TolAbs");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: RMaxFront " << convertCmToMm(rMaxFront) << "\tRmaxBack "
                                 << convertCmToMm(rMaxBack) << "\tTrims " << convertCmToMm(trimLeft) << ":"
                                 << convertCmToMm(trimRight) << "\tTolAbs " << convertCmToMm(tolAbs);
#endif
    slopeBot = args.value<double>("SlopeBottom");
    slopeTop = args.value<double>("SlopeTop");
    slopeTopF = args.value<double>("SlopeTopFront");
    modType = args.value<int>("ModType");
    modNumber = args.value<int>("ModNumber");
    layerType = args.value<int>("LayerType");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: slopeBot " << slopeBot << "\tslopeTop " << slopeTop
                                 << "\tslopeTopF " << slopeTopF << "\tmodType " << modType << "\tmodNumber "
                                 << modNumber << "\tlayerType " << layerType;
#endif
    layerNumber = args.value<std::vector<int> >("LayerNumber");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << layerNumber.size() << " layer Numbers";
    for (unsigned int i = 0; i < layerNumber.size(); ++i)
      edm::LogVerbatim("HCalGeom") << "LayerNumber[" << i << "] = " << layerNumber[i];
#endif
    phiName = args.value<std::vector<std::string> >("PhiName");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << phiName.size() << " phi sectors";
    for (unsigned int i = 0; i < phiName.size(); ++i)
      edm::LogVerbatim("HCalGeom") << "PhiName[" << i << "] = " << phiName[i];
#endif
    layerName = args.value<std::vector<std::string> >("LayerName");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << layerName.size() << " layers";
    for (unsigned int i = 0; i < layerName.size(); ++i)
      edm::LogVerbatim("HCalGeom") << "LayerName[" << i << "] = " << layerName[i];
#endif
    idName = args.value<std::string>("MotherName");
    idOffset = args.value<int>("IdOffset");
    modName = args.value<std::string>("ModName");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: Parent " << args.parentName() << "   " << modName
                                 << " idName " << idName << " NameSpace " << ns.name() << " Offset " << idOffset;
#endif

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "==>> Constructing DDHCalEndcapModuleAlgo...";
#endif

    dd4hep::Volume mother = ns.volume(args.parentName());
    if (modType == 0)
      constructInsideModule0(ctxt, e, mother);
    else
      constructInsideModule(ctxt, e, mother);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "<<== End of DDHCalEndcapModuleAlgo construction ...";
#endif
  }

  void constructInsideModule0(cms::DDParsingContext& ctxt, xml_h e, dd4hep::Volume& module) {
    cms::DDNamespace ns(ctxt, e, true);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: \t\tInside module0";
#endif
    ///////////////////////////////////////////////////////////////
    //Pointers to the Materials
    dd4hep::Material matabsorbr = ns.material(absorberMat);
    dd4hep::Material matplastic = ns.material(plasticMat);
    dd4hep::Rotation3D rot = getRotation(rotstr, ns);

    int layer = layerNumber[0];
    int layer0 = layerNumber[1];
    std::string name;
    dd4hep::Solid solid;
    dd4hep::Volume glog;
    for (unsigned int iphi = 0; iphi < phiName.size(); iphi++) {
      HCalEndcapModuleAlgo::HcalEndcapPar parm = parameterLayer0(iphi);
      name = idName + modName + layerName[0] + phiName[iphi];
      solid = dd4hep::Trap(ns.prepend(name),
                           0.5 * layerThick,
                           0,
                           0,
                           parm.yh1,
                           parm.bl1,
                           parm.tl1,
                           parm.alp,
                           parm.yh2,
                           parm.bl1,
                           parm.tl2,
                           parm.alp);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << solid.name() << " Trap made of " << plasticMat
                                   << " of dimensions " << convertCmToMm(0.5 * layerThick) << ", 0, 0, "
                                   << convertCmToMm(parm.yh1) << ", " << convertCmToMm(parm.bl1) << ", "
                                   << convertCmToMm(parm.tl1) << ", " << convertRadToDeg(parm.alp) << ", "
                                   << convertCmToMm(parm.yh2) << ", " << convertCmToMm(parm.bl2) << ", "
                                   << convertCmToMm(parm.tl2) << ", " << convertRadToDeg(parm.alp);
#endif
      glog = dd4hep::Volume(solid.name(), solid, matplastic);

      dd4hep::Position r1(parm.xpos, parm.ypos, parm.zpos);
      module.placeVolume(glog, idOffset + layer + 1, dd4hep::Transform3D(rot, r1));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << glog.name() << " number " << (idOffset + layer + 1)
                                   << " positioned in " << module.name() << " at (" << convertCmToMm(parm.xpos) << ", "
                                   << convertCmToMm(parm.ypos) << ", " << convertCmToMm(parm.zpos)
                                   << ") with rotation: " << rot;
#endif
      //Now construct the layer of scintillator inside this
      int copyNo = layer0 * 10 + layerType;
      name = modName + layerName[0] + phiName[iphi];
      constructScintLayer(glog, scintThick, parm, name, copyNo, ns);
    }

    //Now the absorber layer
    double zi = zMinBlock + layerThick;
    double zo = zi + 0.5 * dzStep;
    double rinF, routF, rinB, routB;
    if (modNumber == 0) {
      rinF = zi * slopeTopF;
      routF = (zi - z1Beam) * slopeTop;
      rinB = zo * slopeTopF;
      routB = (zo - z1Beam) * slopeTop;
    } else if (modNumber > 0) {
      rinF = zi * slopeBot;
      routF = zi * slopeTopF;
      rinB = zo * slopeBot;
      routB = zo * slopeTopF;
    } else {
      rinF = zi * slopeBot;
      routF = (zi - z1Beam) * slopeTop;
      rinB = zo * slopeBot;
      routB = (zo - z1Beam) * slopeTop;
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: Front " << convertCmToMm(zi) << ", " << convertCmToMm(rinF)
                                 << ", " << convertCmToMm(routF) << " Back " << convertCmToMm(zo) << ", "
                                 << convertCmToMm(rinB) << ", " << convertCmToMm(routB);
#endif
    HCalEndcapModuleAlgo::HcalEndcapPar parm = parameterLayer(0, rinF, routF, rinB, routB, zi, zo);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: Trim " << convertCmToMm(tolAbs) << " Param "
                                 << convertCmToMm(parm.yh1) << ", " << convertCmToMm(parm.bl1) << ", "
                                 << convertCmToMm(parm.tl1) << ", " << convertCmToMm(parm.yh2) << ", "
                                 << convertCmToMm(parm.bl2) << ", " << convertCmToMm(parm.tl2);
#endif
    parm.bl1 -= tolAbs;
    parm.tl1 -= tolAbs;
    parm.bl2 -= tolAbs;
    parm.tl2 -= tolAbs;

    name = idName + modName + layerName[0] + "Absorber";
    solid = dd4hep::Trap(ns.prepend(name),
                         0.5 * moduleThick,
                         parm.theta,
                         parm.phi,
                         parm.yh1,
                         parm.bl1,
                         parm.tl1,
                         parm.alp,
                         parm.yh2,
                         parm.bl2,
                         parm.tl2,
                         parm.alp);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << solid.name() << " Trap made of " << matabsorbr.name()
                                 << " of dimensions " << convertCmToMm(0.5 * moduleThick) << ", "
                                 << convertRadToDeg(parm.theta) << ", " << convertRadToDeg(parm.phi) << ", "
                                 << convertCmToMm(parm.yh1) << ", " << convertCmToMm(parm.bl1) << ", "
                                 << convertCmToMm(parm.tl1) << ", " << convertRadToDeg(parm.alp) << ", "
                                 << convertCmToMm(parm.yh2) << ", " << convertCmToMm(parm.bl2) << ", "
                                 << convertCmToMm(parm.tl2) << ", " << convertRadToDeg(parm.alp);
#endif
    glog = dd4hep::Volume(solid.name(), solid, matabsorbr);

    dd4hep::Position r2(parm.xpos, parm.ypos, parm.zpos);
    module.placeVolume(glog, idOffset + layer + 1, dd4hep::Transform3D(rot, r2));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << glog.name() << " number 1 positioned in "
                                 << module.name() << " at (" << convertCmToMm(parm.xpos) << ", "
                                 << convertCmToMm(parm.ypos) << ", " << convertCmToMm(parm.zpos)
                                 << ") with rotation: " << rot;
#endif
  }

  void constructInsideModule(cms::DDParsingContext& ctxt, xml_h e, dd4hep::Volume& module) {
    cms::DDNamespace ns(ctxt, e, true);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: \t\tInside module";
#endif
    ///////////////////////////////////////////////////////////////
    //Pointers to the Rotation Matrices and to the Materials
    dd4hep::Material matter = ns.material(genMaterial);
    dd4hep::Material matplastic = ns.material(plasticMat);
    dd4hep::Rotation3D rot = getRotation(rotstr, ns);

    double alpha = (1._pi) / sectors;
    double zi = zMinBlock;

    for (unsigned int i = 0; i < layerName.size(); i++) {
      std::string name;
      dd4hep::Solid solid;
      dd4hep::Volume glog, plog;
      int layer = layerNumber[i];
      double zo = zi + 0.5 * dzStep;

      for (unsigned int iphi = 0; iphi < phiName.size(); iphi++) {
        double ziAir = zo - moduleThick;
        double rinF, rinB;
        if (modNumber == 0) {
          rinF = ziAir * slopeTopF;
          rinB = zo * slopeTopF;
        } else {
          rinF = ziAir * slopeBot;
          rinB = zo * slopeBot;
        }
        double routF = getRout(ziAir);
        double routB = getRout(zo);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: Layer " << i << " Phi " << iphi << " Front "
                                     << convertCmToMm(ziAir) << ", " << convertCmToMm(rinF) << ", "
                                     << convertCmToMm(routF) << " Back " << convertCmToMm(zo) << ", "
                                     << convertCmToMm(rinB) << ", " << convertCmToMm(routB);
#endif
        HCalEndcapModuleAlgo::HcalEndcapPar parm = parameterLayer(iphi, rinF, routF, rinB, routB, ziAir, zo);

        name = idName + modName + layerName[i] + phiName[iphi] + "Air";
        solid = dd4hep::Trap(ns.prepend(name),
                             0.5 * moduleThick,
                             parm.theta,
                             parm.phi,
                             parm.yh1,
                             parm.bl1,
                             parm.tl1,
                             parm.alp,
                             parm.yh2,
                             parm.bl2,
                             parm.tl2,
                             parm.alp);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << solid.name() << " Trap made of " << matter.name()
                                     << " of dimensions " << convertCmToMm(0.5 * moduleThick) << ", "
                                     << convertRadToDeg(parm.theta) << ", " << convertRadToDeg(parm.phi) << ", "
                                     << convertCmToMm(parm.yh1) << ", " << convertCmToMm(parm.bl1) << ", "
                                     << convertCmToMm(parm.tl1) << ", " << convertRadToDeg(parm.alp) << ", "
                                     << convertCmToMm(parm.yh2) << ", " << convertCmToMm(parm.bl2) << ", "
                                     << convertCmToMm(parm.tl2) << ", " << convertRadToDeg(parm.alp);
#endif
        glog = dd4hep::Volume(solid.name(), solid, matter);

        dd4hep::Position r1(parm.xpos, parm.ypos, parm.zpos);
        module.placeVolume(glog, layer + 1, dd4hep::Transform3D(rot, r1));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << glog.name() << " number " << layer + 1
                                     << " positioned in " << module.name() << " at (" << convertCmToMm(parm.xpos)
                                     << ", " << convertCmToMm(parm.ypos) << ", " << convertCmToMm(parm.zpos)
                                     << ") with rotation: " << rot;
#endif
        //Now the plastic with scintillators
        parm.yh1 = 0.5 * (routF - rinB) - getTrim(iphi);
        parm.bl1 = 0.5 * rinB * tan(alpha) - getTrim(iphi);
        parm.tl1 = 0.5 * routF * tan(alpha) - getTrim(iphi);
        name = idName + modName + layerName[i] + phiName[iphi];
        solid = dd4hep::Trap(ns.prepend(name),
                             0.5 * layerThick,
                             0,
                             0,
                             parm.yh1,
                             parm.bl1,
                             parm.tl1,
                             parm.alp,
                             parm.yh1,
                             parm.bl1,
                             parm.tl1,
                             parm.alp);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << solid.name() << " Trap made of "
                                     << matplastic.name() << " of dimensions " << convertCmToMm(0.5 * layerThick)
                                     << ", 0, 0, " << convertCmToMm(parm.yh1) << ", " << convertCmToMm(parm.bl1) << ", "
                                     << convertCmToMm(parm.tl1) << ", " << convertRadToDeg(parm.alp) << ", "
                                     << convertCmToMm(parm.yh1) << ", " << convertCmToMm(parm.bl1) << ", "
                                     << convertCmToMm(parm.tl1) << ", " << convertRadToDeg(parm.alp);
#endif
        plog = dd4hep::Volume(solid.name(), solid, matplastic);

        double ypos = 0.5 * (routF + rinB) - parm.xpos;
        dd4hep::Position r2(0., ypos, 0.);
        glog.placeVolume(plog, idOffset + layer + 1, r2);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << plog.name() << " number "
                                     << (idOffset + layer + 1) << " positioned in " << glog.name() << " at (0, "
                                     << convertCmToMm(ypos) << ", 0) with no rotation";
#endif
        //Constructin the scintillators inside
        int copyNo = layer * 10 + layerType;
        name = modName + layerName[i] + phiName[iphi];
        constructScintLayer(plog, scintThick, parm, name, copyNo, ns);
        zo += 0.5 * dzStep;
      }  // End of loop over phi indices
      zi = zo - 0.5 * dzStep;
    }  // End of loop on layers
  }

  HcalEndcapPar parameterLayer0(unsigned int iphi) {
    HCalEndcapModuleAlgo::HcalEndcapPar parm;
    //Given module and layer number compute parameters of trapezoid
    //and positioning parameters
    double alpha = (1._pi) / sectors;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "Input " << iphi << " Alpha " << convertRadToDeg(alpha);
#endif
    double zi, zo;
    if (iphi == 0) {
      zi = zMinBlock;
      zo = zi + layerThick;
    } else {
      zo = zMaxBlock;
      zi = zo - layerThick;
    }
    double rin, rout;
    if (modNumber == 0) {
      rin = zo * slopeTopF;
      rout = (zi - z1Beam) * slopeTop;
    } else if (modNumber > 0) {
      rin = zo * slopeBot;
      rout = zi * slopeTopF;
    } else {
      rin = zo * slopeBot;
      rout = (zi - z1Beam) * slopeTop;
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "ModNumber " << modNumber << " " << convertCmToMm(zi) << " " << convertCmToMm(zo)
                                 << " " << slopeTopF << " " << slopeTop << " " << slopeBot << " " << convertCmToMm(rin)
                                 << " " << convertCmToMm(rout) << " " << convertCmToMm(getTrim(iphi));
#endif
    double yh = 0.5 * (rout - rin);
    double bl = 0.5 * rin * tan(alpha);
    double tl = 0.5 * rout * tan(alpha);
    parm.xpos = 0.5 * (rin + rout);
    parm.ypos = 0.5 * (bl + tl);
    parm.zpos = 0.5 * (zi + zo);
    parm.yh1 = parm.yh2 = yh - getTrim(iphi);
    parm.bl1 = parm.bl2 = bl - getTrim(iphi);
    parm.tl1 = parm.tl2 = tl - getTrim(iphi);
    parm.alp = atan(0.5 * tan(alpha));
    if (iphi == 0) {
      parm.ypos = -parm.ypos;
    } else {
      parm.alp = -parm.alp;
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "Output Dimensions " << convertCmToMm(parm.yh1) << " " << convertCmToMm(parm.bl1)
                                 << " " << convertCmToMm(parm.tl1) << " " << convertRadToDeg(parm.alp) << " Position "
                                 << convertCmToMm(parm.xpos) << " " << convertCmToMm(parm.ypos) << " "
                                 << convertCmToMm(parm.zpos);
#endif
    return parm;
  }

  HcalEndcapPar parameterLayer(
      unsigned int iphi, double rinF, double routF, double rinB, double routB, double zi, double zo) {
    HCalEndcapModuleAlgo::HcalEndcapPar parm;
    //Given rin, rout compute parameters of the trapezoid and
    //position of the trapezoid for a standrd layer
    double alpha = (1._pi) / sectors;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "Input " << iphi << " Front " << convertCmToMm(rinF) << " " << convertCmToMm(routF)
                                 << " " << convertCmToMm(zi) << " Back " << convertCmToMm(rinB) << " "
                                 << convertCmToMm(routB) << " " << convertCmToMm(zo) << " Alpha "
                                 << convertRadToDeg(alpha);
#endif
    parm.yh1 = 0.5 * (routF - rinB);
    parm.bl1 = 0.5 * rinB * tan(alpha);
    parm.tl1 = 0.5 * routF * tan(alpha);
    parm.yh2 = 0.5 * (routF - rinB);
    parm.bl2 = 0.5 * rinB * tan(alpha);
    parm.tl2 = 0.5 * routF * tan(alpha);
    double dx = 0.25 * (parm.bl2 + parm.tl2 - parm.bl1 - parm.tl1);
    double dy = 0.5 * (rinB + routF - rinB - routF);
    parm.xpos = 0.25 * (rinB + routF + rinB + routF);
    parm.ypos = 0.25 * (parm.bl2 + parm.tl2 + parm.bl1 + parm.tl1);
    parm.zpos = 0.5 * (zi + zo);
    parm.alp = atan(0.5 * tan(alpha));
    if (iphi == 0) {
      parm.ypos = -parm.ypos;
    } else {
      parm.alp = -parm.alp;
      dx = -dx;
    }
    double r = sqrt(dx * dx + dy * dy);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "dx|dy|r " << convertCmToMm(dx) << ":" << convertCmToMm(dy) << ":"
                                 << convertCmToMm(r);
#endif
    if (r > 1.0e-8) {
      parm.theta = atan(r / (zo - zi));
      parm.phi = atan2(dy, dx);
    } else {
      parm.theta = parm.phi = 0;
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "Output Dimensions " << convertCmToMm(parm.yh1) << " " << convertCmToMm(parm.bl1)
                                 << " " << convertCmToMm(parm.tl1) << " " << convertCmToMm(parm.yh2) << " "
                                 << convertCmToMm(parm.bl2) << " " << convertCmToMm(parm.tl2) << " "
                                 << convertRadToDeg(parm.alp) << " " << convertRadToDeg(parm.theta) << " "
                                 << convertRadToDeg(parm.phi) << " Position " << convertCmToMm(parm.xpos) << " "
                                 << convertCmToMm(parm.ypos) << " " << convertCmToMm(parm.zpos);
#endif
    return parm;
  }

  void constructScintLayer(dd4hep::Volume& detector,
                           double dz,
                           HCalEndcapModuleAlgo::HcalEndcapPar parm,
                           const std::string& nm,
                           int id,
                           cms::DDNamespace& ns) {
    dd4hep::Material matter = ns.material(scintMat);
    std::string name = idName + "Scintillator" + nm;

    dd4hep::Solid solid = dd4hep::Trap(
        ns.prepend(name), 0.5 * dz, 0, 0, parm.yh1, parm.bl1, parm.tl1, parm.alp, parm.yh1, parm.bl1, parm.tl1, parm.alp);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << solid.name() << " Trap made of " << scintMat
                                 << " of dimensions " << convertCmToMm(0.5 * dz) << ", 0, 0, "
                                 << convertCmToMm(parm.yh1) << ", " << convertCmToMm(parm.bl1) << ", "
                                 << convertCmToMm(parm.tl1) << ", " << convertRadToDeg(parm.alp) << ", "
                                 << convertCmToMm(parm.yh1) << ", " << convertCmToMm(parm.bl1) << ", "
                                 << convertCmToMm(parm.tl1) << ", " << convertRadToDeg(parm.alp);
#endif
    dd4hep::Volume glog(solid.name(), solid, matter);

    detector.placeVolume(glog, id);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << glog.name() << " number " << id << " positioned in "
                                 << detector.name() << " at (0,0,0) with no rotation";
#endif
  }

  double getTrim(unsigned int j) const {
    if (j == 0)
      return trimLeft;
    else
      return trimRight;
  }

  double getRout(double z) const {
    double r = (modNumber >= 0) ? ((z - z1Beam) * slopeTop) : z * slopeTopF;
    if (z > ziDip) {
      if (r > rMaxBack)
        r = rMaxBack;
    } else {
      if (r > rMaxFront)
        r = rMaxFront;
    }
    return r;
  }

  dd4hep::Rotation3D getRotation(const std::string& rotstr, cms::DDNamespace& ns) {
    std::string rot = (strchr(rotstr.c_str(), NAMESPACE_SEP) == nullptr) ? ("rotations:" + rotstr) : rotstr;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "getRotation: " << rotstr << ":" << rot << ":" << ns.rotation(rot);
#endif
    return ns.rotation(rot);
  }
};

static long algorithm(dd4hep::Detector& /* description */,
                      cms::DDParsingContext& ctxt,
                      xml_h e,
                      dd4hep::SensitiveDetector& /* sens */) {
  HCalEndcapModuleAlgo hcalendcapalgo(ctxt, e);
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hcal_DDHCalEndcapModuleAlgo, algorithm)
