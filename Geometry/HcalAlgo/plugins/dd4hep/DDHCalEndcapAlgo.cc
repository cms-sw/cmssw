///////////////////////////////////////////////////////////////////////////////
// File: DDHCalEndcapAlgo.cc
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
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

struct HCalEndcapAlgo {
  std::string genMaterial;             //General material
  int nsectors;                        //Number of potenital straight edges
  int nsectortot;                      //Number of straight edges (actual)
  int nEndcap;                         //Number of endcaps
  std::vector<int> eModule;            //Modules to be present in part i (?)
  std::string rotHalf;                 //Rotation matrix for half
  std::string rotation;                //Rotation matrix to place in mother
  double zShift;                       //needed for TB setup (move HE)
  double zFront;                       //Z of the front section
  double zEnd;                         //Outer Z of the HE
  double ziNose;                       //Starting Z of the nose
  double ziL0Nose;                     //Starting Z of layer 0 at nose
  double ziBody;                       //Starting Z of the body
  double ziL0Body;                     //Starting Z of layer 0 at body
  double ziKink;                       //Position of the kink point
  double z0Beam;                       //Position of gap front along z-axis
  double z1Beam;                       //Position of gap end   along z-axis
  double ziDip;                        //Starting Z of dipped part of body
  double dzStep;                       //Width in Z of a layer
  double dzShift;                      //Shift in Z for HE
  double zShiftHac2;                   //needed for TB (remove part Hac2)
  double rout;                         //Outer R of the HE
  double riKink;                       //Inner radius at kink point
  double riDip;                        //Inner radius at the dip point
  double roDip;                        //Outer radius at the dip point
  double heboxDepth;                   //Depth of the HE box
  double drEnd;                        //Shift in R for the end absorber
  double angTop;                       //Angle of top end of HE
  double angBot;                       //Angle of the bottom end of HE
  double angGap;                       //Gap angle (in degrees)
  double slope;                        //Slope of the gap on HE side
  std::string absMat;                  //Absorber     material
  int modules;                         //Number of modules
  std::vector<std::string> modName;    //Name
  std::vector<std::string> modMat;     //Material
  std::vector<int> modType;            //Type (0/1 for front/standard)
  std::vector<int> sectionModule;      //Number of sections in a module
  std::vector<int> layerN;             //Number of layers
  std::vector<int> layerN0;            //Layer numbers in section 0
  std::vector<int> layerN1;            //Layer numbers in section 1
  std::vector<int> layerN2;            //Layer numbers in section 2
  std::vector<int> layerN3;            //Layer numbers in section 3
  std::vector<int> layerN4;            //Layer numbers in section 4
  std::vector<int> layerN5;            //Layer numbers in section 5
  std::vector<double> thick;           //Thickness of absorber/air
  std::vector<double> trimLeft;        //Trimming of left  layers in module
  std::vector<double> trimRight;       //Trimming of right layers in module
  std::vector<double> zminBlock;       //Minimum Z
  std::vector<double> zmaxBlock;       //Maximum Z
  std::vector<double> rinBlock1;       //Inner Radius
  std::vector<double> routBlock1;      //Outer Radius at zmin
  std::vector<double> rinBlock2;       //Inner Radius
  std::vector<double> routBlock2;      //Outer Radius at zmax
  int phiSections;                     //Number of phi sections
  std::vector<std::string> phiName;    //Name of Phi sections
  int layers;                          //Number of layers
  std::vector<std::string> layerName;  //Layer Names
  std::vector<int> layerType;          //Detector type in each layer
  std::vector<double> layerT;          //Layer thickness (plastic + scint.)
  std::vector<double> scintT;          //Scintillator thickness
  std::string plastMat;                //Plastic      material
  std::string scintMat;                //Scintillator material
  std::string rotmat;                  //Rotation matrix for positioning
  std::string idName;                  //Name of the "parent" volume.
  int idOffset;                        // Geant4 ID's...    = 4000;
  double tolPos, tolAbs;               //Tolerances

  HCalEndcapAlgo() = delete;

  HCalEndcapAlgo(cms::DDParsingContext& ctxt, xml_h e) {
    cms::DDNamespace ns(ctxt, e, true);
    cms::DDAlgoArguments args(ctxt, e);

    genMaterial = args.value<std::string>("MaterialName");
    rotation = args.value<std::string>("Rotation");
    nsectors = args.value<int>("Sector");
    nsectortot = args.value<int>("SectorTot");
    nEndcap = args.value<int>("Endcap");
    rotHalf = args.value<std::string>("RotHalf");
    zShift = args.value<double>("ZShift");
    zFront = args.value<double>("ZFront");
    zEnd = args.value<double>("ZEnd");
    ziNose = args.value<double>("ZiNose");
    ziL0Nose = args.value<double>("ZiL0Nose");
    ziBody = args.value<double>("ZiBody");
    ziL0Body = args.value<double>("ZiL0Body");
    z0Beam = args.value<double>("Z0Beam");
    ziDip = args.value<double>("ZiDip");
    dzStep = args.value<double>("DzStep");
    zShiftHac2 = args.value<double>("ZShiftHac2");
    double gap = args.value<double>("Gap");
    double z1 = args.value<double>("Z1");
    double r1 = args.value<double>("R1");
    rout = args.value<double>("Rout");
    heboxDepth = args.value<double>("HEboxDepth");
    drEnd = args.value<double>("DrEnd");
    double etamin = args.value<double>("Etamin");
    angBot = args.value<double>("AngBot");
    angGap = args.value<double>("AngGap");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: General material " << genMaterial << "\tSectors " << nsectors
                                 << ",  " << nsectortot << "\tEndcaps " << nEndcap << "\tRotation matrix for half "
                                 << rotHalf << "\n\tzFront " << convertCmToMm(zFront) << " zEnd " << convertCmToMm(zEnd)
                                 << " ziNose " << convertCmToMm(ziNose) << " ziL0Nose " << convertCmToMm(ziL0Nose)
                                 << " ziBody " << convertCmToMm(ziBody) << " ziL0Body " << convertCmToMm(ziL0Body)
                                 << " z0Beam " << convertCmToMm(z0Beam) << " ziDip " << convertCmToMm(ziDip)
                                 << " dzStep " << convertCmToMm(dzStep) << " Gap " << convertCmToMm(gap) << " z1 "
                                 << convertCmToMm(z1) << "\n\tr1 " << convertCmToMm(r1) << " rout "
                                 << convertCmToMm(rout) << " HeboxDepth " << convertCmToMm(heboxDepth) << " drEnd "
                                 << convertCmToMm(drEnd) << "\tetamin " << etamin << " Bottom angle " << angBot
                                 << " Gap angle " << angGap << " Z-Shift " << convertCmToMm(zShift) << " "
                                 << convertCmToMm(zShiftHac2);
#endif

    //Derived quantities
    angTop = 2.0 * atan(exp(-etamin));
    slope = tan(angGap);
    z1Beam = z1 - r1 / slope;
    ziKink = z1Beam + rout / slope;
    riKink = ziKink * tan(angBot);
    riDip = ziDip * tan(angBot);
    roDip = rout - heboxDepth;
    dzShift = (z1Beam - z0Beam) - gap / sin(angGap);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: angTop " << convertRadToDeg(angTop) << "\tSlope " << slope
                                 << "\tDzShift " << convertCmToMm(dzShift) << "\n\tz1Beam " << convertCmToMm(z1Beam)
                                 << "\tziKink" << convertCmToMm(ziKink) << "\triKink " << convertCmToMm(riKink)
                                 << "\triDip " << convertCmToMm(riDip) << "\n\troDip " << convertCmToMm(roDip)
                                 << "\tRotation " << rotation;
#endif

    ///////////////////////////////////////////////////////////////
    //Modules
    absMat = args.value<std::string>("AbsMat");
    modules = args.value<int>("Modules");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Number of modules " << modules << " and absorber material "
                                 << absMat;
#endif

    modName = args.value<std::vector<std::string> >("ModuleName");
    modMat = args.value<std::vector<std::string> >("ModuleMat");
    modType = args.value<std::vector<int> >("ModuleType");
    sectionModule = args.value<std::vector<int> >("SectionModule");
    thick = args.value<std::vector<double> >("ModuleThick");
    trimLeft = args.value<std::vector<double> >("TrimLeft");
    trimRight = args.value<std::vector<double> >("TrimRight");
    eModule = args.value<std::vector<int> >("EquipModule");
    layerN = args.value<std::vector<int> >("LayerN");
    layerN0 = args.value<std::vector<int> >("LayerN0");
    layerN1 = args.value<std::vector<int> >("LayerN1");
    layerN2 = args.value<std::vector<int> >("LayerN2");
    layerN3 = args.value<std::vector<int> >("LayerN3");
    layerN4 = args.value<std::vector<int> >("LayerN4");
    layerN5 = args.value<std::vector<int> >("LayerN5");
#ifdef EDM_ML_DEBUG
    for (int i = 0; i < modules; i++) {
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << modName[i] << " type " << modType[i] << " Sections "
                                   << sectionModule[i] << " thickness of absorber/air " << convertCmToMm(thick[i])
                                   << " trim " << convertCmToMm(trimLeft[i]) << ", " << convertCmToMm(trimRight[i])
                                   << " equip module " << eModule[i] << " with " << layerN[i] << " layers";
      if (i == 0) {
        for (int j = 0; j < layerN[i]; j++) {
          edm::LogVerbatim("HCalGeom") << "\t " << layerN0[j] << "/" << layerN0[j + 1];
        }
      } else if (i == 1) {
        for (int j = 0; j < layerN[i]; j++) {
          edm::LogVerbatim("HCalGeom") << "\t " << layerN1[j] << "/" << layerN1[j + 1];
        }
      } else if (i == 2) {
        for (int j = 0; j < layerN[i]; j++) {
          edm::LogVerbatim("HCalGeom") << "\t " << layerN2[j];
        }
      } else if (i == 3) {
        for (int j = 0; j < layerN[i]; j++) {
          edm::LogVerbatim("HCalGeom") << "\t " << layerN3[j];
        }
      } else if (i == 4) {
        for (int j = 0; j < layerN[i]; j++) {
          edm::LogVerbatim("HCalGeom") << "\t " << layerN4[j];
        }
      } else if (i == 5) {
        for (int j = 0; j < layerN[i]; j++) {
          edm::LogVerbatim("HCalGeom") << "\t " << layerN5[j];
        }
      }
    }
#endif

    ///////////////////////////////////////////////////////////////
    //Layers
    phiSections = args.value<int>("PhiSections");
    phiName = args.value<std::vector<std::string> >("PhiName");
    layers = args.value<int>("Layers");
    layerName = args.value<std::vector<std::string> >("LayerName");
    layerType = args.value<std::vector<int> >("LayerType");
    layerT = args.value<std::vector<double> >("LayerT");
    scintT = args.value<std::vector<double> >("ScintT");
    scintMat = args.value<std::string>("ScintMat");
    plastMat = args.value<std::string>("PlastMat");
    rotmat = args.value<std::string>("RotMat");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Phi Sections " << phiSections;
    for (int i = 0; i < phiSections; i++)
      edm::LogVerbatim("HCalGeom") << "\tName[" << i << "] : " << phiName[i];
    edm::LogVerbatim("HCalGeom") << "\tPlastic: " << plastMat << "\tScintillator: " << scintMat << "\tRotation matrix "
                                 << rotmat << "\n\tNumber of layers " << layers;
    for (int i = 0; i < layers; i++) {
      edm::LogVerbatim("HCalGeom") << "\t" << layerName[i] << "\tType " << layerType[i] << "\tThickness "
                                   << convertCmToMm(layerT[i]) << "\tScint.Thick " << convertCmToMm(scintT[i]);
    }
#endif

    ///////////////////////////////////////////////////////////////
    // Derive bounding of the modules
    int module = 0;
    // Layer 0 (Nose)
    if (modules > 0) {
      zminBlock.emplace_back(ziL0Nose);
      zmaxBlock.emplace_back(zminBlock[module] + layerT[0] + 0.5 * dzStep);
      rinBlock1.emplace_back(zminBlock[module] * tan(angTop));
      rinBlock2.emplace_back(zmaxBlock[module] * tan(angTop));
      routBlock1.emplace_back((zminBlock[module] - z1Beam) * slope);
      routBlock2.emplace_back((zmaxBlock[module] - z1Beam) * slope);
      module++;
    }
    // Layer 0 (Body)
    if (modules > 1) {
      zminBlock.emplace_back(ziL0Body);
      zmaxBlock.emplace_back(zminBlock[module] + layerT[0] + 0.5 * dzStep);
      rinBlock1.emplace_back(zminBlock[module] * tan(angBot));
      rinBlock2.emplace_back(zmaxBlock[module] * tan(angBot));
      routBlock1.emplace_back(zminBlock[module] * tan(angTop));
      routBlock2.emplace_back(zmaxBlock[module] * tan(angTop));
      module++;
    }
    // Hac1
    if (modules > 2) {
      zminBlock.emplace_back(ziNose);
      zmaxBlock.emplace_back(ziBody);
      rinBlock1.emplace_back(zminBlock[module] * tan(angTop));
      rinBlock2.emplace_back(zmaxBlock[module] * tan(angTop));
      routBlock1.emplace_back((zminBlock[module] - z1Beam) * slope);
      routBlock2.emplace_back((zmaxBlock[module] - z1Beam) * slope);
      module++;
    }
    // Hac2
    if (modules > 3) {
      zminBlock.emplace_back(ziBody);
      zmaxBlock.emplace_back(zminBlock[module] + layerN[3] * dzStep);
      rinBlock1.emplace_back(zminBlock[module] * tan(angBot));
      rinBlock2.emplace_back(zmaxBlock[module] * tan(angBot));
      routBlock1.emplace_back((zmaxBlock[module - 1] - z1Beam) * slope);
      routBlock2.emplace_back(rout);
      module++;
    }
    // Hac3
    if (modules > 4) {
      zminBlock.emplace_back(zmaxBlock[module - 1]);
      zmaxBlock.emplace_back(zminBlock[module] + layerN[4] * dzStep);
      rinBlock1.emplace_back(zminBlock[module] * tan(angBot));
      rinBlock2.emplace_back(zmaxBlock[module] * tan(angBot));
      routBlock1.emplace_back(rout);
      routBlock2.emplace_back(rout);
      module++;
    }
    // Hac4
    if (modules > 5) {
      zminBlock.emplace_back(zmaxBlock[module - 1]);
      zmaxBlock.emplace_back(zminBlock[module] + layerN[5] * dzStep);
      rinBlock1.emplace_back(zminBlock[module] * tan(angBot));
      rinBlock2.emplace_back(zmaxBlock[module] * tan(angBot));
      routBlock1.emplace_back(rout);
      routBlock2.emplace_back(roDip);
      module++;
    }
#ifdef EDM_ML_DEBUG
    for (int i = 0; i < module; i++)
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Module " << i << "\tZ/Rin/Rout " << convertCmToMm(zminBlock[i])
                                   << ", " << convertCmToMm(zmaxBlock[i]) << "/ " << convertCmToMm(rinBlock1[i]) << ", "
                                   << convertCmToMm(rinBlock2[i]) << "/ " << convertCmToMm(routBlock1[i]) << ", "
                                   << convertCmToMm(routBlock2[i]);
#endif

    idName = args.value<std::string>("MotherName");
    idOffset = args.value<int>("IdOffset");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Parent " << args.parentName() << " idName " << idName
                                 << " NameSpace " << ns.name() << " Offset " << idOffset;
#endif

    tolPos = args.value<double>("TolPos");
    tolAbs = args.value<double>("TolAbs");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Tolerances - Positioning " << convertCmToMm(tolPos)
                                 << " Absorber " << convertCmToMm(tolAbs);
    edm::LogVerbatim("HCalGeom") << "==>> Constructing DDHCalEndcapAlgo...";
#endif

    dd4hep::Volume parent = ns.volume(args.parentName());
    constructGeneralVolume(ns, parent);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "<<== End of DDHCalEndcapAlgo construction ...";
#endif
  }

  void constructGeneralVolume(cms::DDNamespace& ns, dd4hep::Volume& parent) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: General volume...";
#endif

    bool proto = true;
    for (int i = 0; i < 3; i++)
      if (eModule[i] > 0)
        proto = false;

    dd4hep::Rotation3D rot = getRotation(rotation, ns);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << " Rotation matrix " << rotation << " Rotation " << rot;
#endif

    dd4hep::Position r0(0, 0, zShift);
    double alpha = (1._pi) / nsectors;
    double dphi = nsectortot * (2._pi) / nsectors;

    //!!!!!!!!!!!!!!!!!Should be zero. And removed as soon as
    //vertical walls are allowed in SolidPolyhedra
    double delz = 0;

    std::vector<double> pgonZ, pgonRmin, pgonRmax;
    if (proto) {
      double zf = ziBody + zShiftHac2;
      pgonZ.emplace_back(zf - dzShift);
      pgonRmin.emplace_back(zf * tan(angBot));
      pgonRmax.emplace_back((zf - z1Beam) * slope);
    } else {
      pgonZ.emplace_back(zFront - dzShift);
      pgonRmin.emplace_back(zFront * tan(angTop));
      pgonRmax.emplace_back((zFront - z1Beam) * slope);
      pgonZ.emplace_back(ziL0Body - dzShift);
      pgonRmin.emplace_back(ziL0Body * tan(angTop));
      pgonRmax.emplace_back((ziL0Body - z1Beam) * slope);
      pgonZ.emplace_back(ziL0Body - dzShift);
      pgonRmin.emplace_back(ziL0Body * tan(angBot));
      pgonRmax.emplace_back((ziL0Body - z1Beam) * slope);
    }
    pgonZ.emplace_back(ziKink - dzShift);
    pgonRmin.emplace_back(riKink);
    pgonRmax.emplace_back(rout);
    pgonZ.emplace_back(ziDip - dzShift);
    pgonRmin.emplace_back(riDip);
    pgonRmax.emplace_back(rout);
    pgonZ.emplace_back(ziDip - dzShift + delz);
    pgonRmin.emplace_back(riDip);
    pgonRmax.emplace_back(roDip);
    pgonZ.emplace_back(zEnd - dzShift);
    pgonRmin.emplace_back(zEnd * tan(angBot));
    pgonRmax.emplace_back(roDip);
    pgonZ.emplace_back(zEnd);
    pgonRmin.emplace_back(zEnd * tan(angBot));
    pgonRmax.emplace_back(roDip);

    std::string name("Null");
    dd4hep::Solid solid = dd4hep::Polyhedra(ns.prepend(idName), nsectortot, -alpha, dphi, pgonZ, pgonRmin, pgonRmax);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << solid.name() << " Polyhedra made of " << genMaterial
                                 << " with " << nsectortot << " sectors from " << convertRadToDeg(-alpha) << " to "
                                 << convertRadToDeg(-alpha + dphi) << " and with " << pgonZ.size() << " sections";
    for (unsigned int i = 0; i < pgonZ.size(); i++)
      edm::LogVerbatim("HCalGeom") << "\t\tZ = " << convertCmToMm(pgonZ[i]) << "\tRmin = " << convertCmToMm(pgonRmin[i])
                                   << "\tRmax = " << convertCmToMm(pgonRmax[i]);
#endif
    dd4hep::Material matter = ns.material(genMaterial);
    dd4hep::Volume genlogic(solid.name(), solid, matter);

    parent.placeVolume(genlogic, 1, dd4hep::Transform3D(rot, r0));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << genlogic.name() << " number 1 positioned in "
                                 << parent.name() << " at (0, 0, " << convertCmToMm(zShift)
                                 << ") with rotation: " << rot;
#endif

    if (nEndcap != 1) {
      rot = getRotation(rotHalf, ns);
      parent.placeVolume(genlogic, 2, dd4hep::Transform3D(rot, r0));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << genlogic.name() << " number 2 "
                                   << "positioned in " << parent.name() << " at (0, 0, " << convertCmToMm(zShift)
                                   << ") with rotation: " << rot;
#endif
    }

    //Forward half
    name = idName + "Front";
    std::vector<double> pgonZMod, pgonRminMod, pgonRmaxMod;
    for (unsigned int i = 0; i < (pgonZ.size() - 1); i++) {
      pgonZMod.emplace_back(pgonZ[i] + dzShift);
      pgonRminMod.emplace_back(pgonRmin[i]);
      pgonRmaxMod.emplace_back(pgonRmax[i]);
    }
    solid = dd4hep::Polyhedra(ns.prepend(name), nsectortot, -alpha, dphi, pgonZMod, pgonRminMod, pgonRmaxMod);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << solid.name() << " Polyhedra made of " << genMaterial
                                 << " with " << nsectortot << " sectors from " << convertRadToDeg(-alpha) << " to "
                                 << convertRadToDeg(-alpha + dphi) << " and with " << pgonZMod.size() << " sections ";
    for (unsigned int i = 0; i < pgonZMod.size(); i++)
      edm::LogVerbatim("HCalGeom") << "\t\tZ = " << convertCmToMm(pgonZMod[i])
                                   << "\tRmin = " << convertCmToMm(pgonRminMod[i])
                                   << "\tRmax = " << convertCmToMm(pgonRmaxMod[i]);
#endif

    dd4hep::Volume genlogich(solid.name(), solid, matter);
    ns.addVolumeNS(genlogich);
    genlogic.placeVolume(genlogich, 1, dd4hep::Position(0, 0, -dzShift));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << genlogich.name() << " number 1 positioned in "
                                 << genlogic.name() << " at (0,0," << -convertCmToMm(dzShift) << ") with no rotation";
#endif

    //Construct sector (from -alpha to +alpha)
    name = idName + "Module";
    solid = dd4hep::Polyhedra(ns.prepend(name), 1, -alpha, 2 * alpha, pgonZMod, pgonRminMod, pgonRmaxMod);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << solid.name() << " Polyhedra made of " << genMaterial
                                 << " with 1 sector from " << convertRadToDeg(-alpha) << " to "
                                 << convertRadToDeg(alpha) << " and with " << pgonZMod.size() << " sections";
    for (unsigned int i = 0; i < pgonZMod.size(); i++)
      edm::LogVerbatim("HCalGeom") << "\t\tZ = " << convertCmToMm(pgonZMod[i])
                                   << "\tRmin = " << convertCmToMm(pgonRminMod[i])
                                   << "\tRmax = " << convertCmToMm(pgonRmaxMod[i]);
#endif

    dd4hep::Volume seclogic(solid.name(), solid, matter);
    for (int ii = 0; ii < nsectortot; ii++) {
      double phi = ii * 2 * alpha;
      dd4hep::Rotation3D rot0;
      if (phi != 0) {
        rot0 = dd4hep::RotationZ(phi);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Creating a new rotation \t 90," << convertRadToDeg(phi)
                                     << ", 90," << convertRadToDeg(phi + 90._deg) << ", 0, 0";
#endif
      }
      genlogich.placeVolume(seclogic, ii + 1, rot0);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << seclogic.name() << " number " << ii + 1
                                   << " positioned in " << genlogich.name() << " at (0, 0, 0) with rotation: " << rot0;
#endif
    }

    //Construct the things inside the sector
    constructInsideSector(ns, seclogic);

    //Backward half
    name = idName + "Back";
    std::vector<double> pgonZBack, pgonRminBack, pgonRmaxBack;
    pgonZBack.emplace_back(zEnd - dzShift);
    pgonRminBack.emplace_back(pgonZBack[0] * tan(angBot) + drEnd);
    pgonRmaxBack.emplace_back(roDip);
    pgonZBack.emplace_back(zEnd);
    pgonRminBack.emplace_back(pgonZBack[1] * tan(angBot) + drEnd);
    pgonRmaxBack.emplace_back(roDip);
    solid = dd4hep::Polyhedra(ns.prepend(name), nsectortot, -alpha, dphi, pgonZBack, pgonRminBack, pgonRmaxBack);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << solid.name() << " Polyhedra made of " << absMat << " with "
                                 << nsectortot << " sectors from " << convertRadToDeg(-alpha) << " to "
                                 << convertRadToDeg(-alpha + dphi) << " and with " << pgonZBack.size() << " sections";
    for (unsigned int i = 0; i < pgonZBack.size(); i++)
      edm::LogVerbatim("HCalGeom") << "\t\tZ = " << convertCmToMm(pgonZBack[i])
                                   << "\tRmin = " << convertCmToMm(pgonRminBack[i])
                                   << "\tRmax = " << convertCmToMm(pgonRmaxBack[i]);
#endif

    dd4hep::Material absMatter = ns.material(absMat);
    dd4hep::Volume glog(solid.name(), solid, absMatter);
    genlogic.placeVolume(glog, 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << glog.name() << " number 1 positioned in " << genlogic.name()
                                 << " at (0,0,0) with no rotation";
#endif
  }

  void constructInsideSector(cms::DDNamespace& ns, dd4hep::Volume& sector) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Modules (" << modules << ") ...";
#endif

    double alpha = (1._pi) / nsectors;
    for (int i = 0; i < modules; i++) {
      std::string name = idName + modName[i];
      dd4hep::Material matter = ns.material(modMat[i]);

      if (eModule[i] > 0) {
        int nsec = sectionModule[i];

        //!!!!!!!!!!!!!!!!!Should be zero. And removed as soon as
        //vertical walls are allowed in SolidPolyhedra
        double deltaz = 0;

        std::vector<double> pgonZ, pgonRmin, pgonRmax;
        if (nsec == 3) {
          double zf = zminBlock[i] + zShiftHac2;
          pgonZ.emplace_back(zf);
          pgonRmin.emplace_back(zf * tan(angBot));
          pgonRmax.emplace_back((zf - z1Beam) * slope);
          pgonZ.emplace_back(ziKink);
          pgonRmin.emplace_back(riKink);
          pgonRmax.emplace_back(rout);
        } else {
          pgonZ.emplace_back(zminBlock[i]);
          pgonRmin.emplace_back(rinBlock1[i]);
          pgonRmax.emplace_back(routBlock1[i]);
        }
        if (nsec == 4) {
          pgonZ.emplace_back(ziDip);
          pgonRmin.emplace_back(riDip);
          pgonRmax.emplace_back(rout);
          pgonZ.emplace_back(pgonZ[1] + deltaz);
          pgonRmin.emplace_back(pgonRmin[1]);
          pgonRmax.emplace_back(roDip);
        }
        pgonZ.emplace_back(zmaxBlock[i]);
        pgonRmin.emplace_back(rinBlock2[i]);
        pgonRmax.emplace_back(routBlock2[i]);

        //Solid & volume
        dd4hep::Solid solid = dd4hep::Polyhedra(ns.prepend(name), 1, -alpha, 2 * alpha, pgonZ, pgonRmin, pgonRmax);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << solid.name() << " Polyhedra made of " << modMat[i]
                                     << " with 1 sector from " << convertRadToDeg(-alpha) << " to "
                                     << convertRadToDeg(alpha) << " and with " << nsec << " sections";
        for (unsigned int k = 0; k < pgonZ.size(); k++)
          edm::LogVerbatim("HCalGeom") << "\t\tZ = " << convertCmToMm(pgonZ[k])
                                       << "\tRmin = " << convertCmToMm(pgonRmin[k])
                                       << "\tRmax = " << convertCmToMm(pgonRmax[k]);
#endif

        dd4hep::Volume glog(solid.name(), solid, matter);

        sector.placeVolume(glog, i + 1);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << glog.name() << " number " << i + 1 << " positioned in "
                                     << sector.name() << " at (0,0,0) with no rotation";
#endif

        if (modType[i] == 0)
          constructInsideModule0(ns, glog, i);
        else
          constructInsideModule(ns, glog, i);
      }
    }
  }

  void constructInsideModule0(cms::DDNamespace& ns, dd4hep::Volume& module, int mod) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: \t\tInside module0 ..." << mod;
#endif

    ///////////////////////////////////////////////////////////////
    //Pointers to the Rotation Matrices and to the Materials
    dd4hep::Rotation3D rot = getRotation(rotmat, ns);
    dd4hep::Material matabsorbr = ns.material(absMat);
    dd4hep::Material matplastic = ns.material(plastMat);

    int layer = getLayer(mod, 0);
    int layer0 = getLayer(mod, 1);
    std::string name;
    double xpos, ypos, zpos;
    dd4hep::Solid solid;
    dd4hep::Volume glog;
    for (int iphi = 0; iphi < phiSections; iphi++) {
      double yh, bl, tl, alp;
      parameterLayer0(mod, layer, iphi, yh, bl, tl, alp, xpos, ypos, zpos);
      name = DDSplit(module.name()).first + layerName[layer] + phiName[iphi];
      solid = dd4hep::Trap(ns.prepend(name), 0.5 * layerT[layer], 0, 0, yh, bl, tl, alp, yh, bl, tl, alp);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << solid.name() << " Trap made of " << plastMat
                                   << " of dimensions " << convertCmToMm(0.5 * layerT[layer]) << ", 0, 0, "
                                   << convertCmToMm(yh) << ", " << convertCmToMm(bl) << ", " << convertCmToMm(tl)
                                   << ", " << convertRadToDeg(alp) << ", " << convertCmToMm(yh) << ", "
                                   << convertCmToMm(bl) << ", " << convertCmToMm(tl) << ", " << convertRadToDeg(alp);
#endif

      glog = dd4hep::Volume(solid.name(), solid, matplastic);

      dd4hep::Position r1(xpos, ypos, zpos);
      module.placeVolume(glog, idOffset + layer + 1, dd4hep::Transform3D(rot, r1));

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << glog.name() << " number " << (idOffset + layer + 1)
                                   << " positioned in " << module.name() << " at (" << convertCmToMm(xpos) << ", "
                                   << convertCmToMm(ypos) << ", " << convertCmToMm(zpos) << " with rotation: " << rot;
#endif

      //Now construct the layer of scintillator inside this
      int copyNo = layer0 * 10 + layerType[layer];
      name = modName[mod] + layerName[layer] + phiName[iphi];
      constructScintLayer(ns, glog, scintT[layer], yh, bl, tl, alp, name, copyNo);
    }

    //Now the absorber layer
    double zi = zminBlock[mod] + layerT[layer];
    double zo = zi + 0.5 * dzStep;
    double rinF, routF, rinB, routB;
    if (mod == 0) {
      rinF = zi * tan(angTop);
      routF = (zi - z1Beam) * slope;
      rinB = zo * tan(angTop);
      routB = (zo - z1Beam) * slope;
    } else {
      rinF = zi * tan(angBot);
      routF = zi * tan(angTop);
      rinB = zo * tan(angBot);
      routB = zo * tan(angTop);
    }

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Module " << mod << " Front " << convertCmToMm(zi) << ", "
                                 << convertCmToMm(rinF) << ", " << convertCmToMm(routF) << " Back " << convertCmToMm(zo)
                                 << ", " << convertCmToMm(rinB) << ", " << convertCmToMm(routB);
#endif

    double yh1, bl1, tl1, yh2, bl2, tl2, theta, phi, alp;
    parameterLayer(
        0, rinF, routF, rinB, routB, zi, zo, yh1, bl1, tl1, yh2, bl2, tl2, alp, theta, phi, xpos, ypos, zpos);
    double fact = tolAbs;

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Trim " << convertCmToMm(fact) << " Param " << convertCmToMm(yh1)
                                 << ", " << convertCmToMm(bl1) << ", " << convertCmToMm(tl1) << ", "
                                 << convertCmToMm(yh2) << ", " << convertCmToMm(bl2) << ", " << convertCmToMm(tl2);
#endif

    bl1 -= fact;
    tl1 -= fact;
    bl2 -= fact;
    tl2 -= fact;

    name = DDSplit(module.name()).first + "Absorber";
    solid = dd4hep::Trap(ns.prepend(name), 0.5 * thick[mod], theta, phi, yh1, bl1, tl1, alp, yh2, bl2, tl2, alp);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << solid.name() << " Trap made of " << absMat
                                 << " of dimensions " << convertCmToMm(0.5 * thick[mod]) << ", "
                                 << convertRadToDeg(theta) << ", " << convertRadToDeg(phi) << ", " << convertCmToMm(yh1)
                                 << ", " << convertCmToMm(bl1) << ", " << convertCmToMm(tl1) << ", "
                                 << convertRadToDeg(alp) << ", " << convertCmToMm(yh2) << ", " << convertCmToMm(bl2)
                                 << ", " << convertCmToMm(tl2) << ", " << convertRadToDeg(alp);
#endif

    glog = dd4hep::Volume(solid.name(), solid, matabsorbr);

    dd4hep::Position r2(xpos, ypos, zpos);
    module.placeVolume(glog, 1, dd4hep::Transform3D(rot, r2));

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << glog.name() << " number 1 positioned in " << module.name()
                                 << " at (" << convertCmToMm(xpos) << ", " << convertCmToMm(ypos) << ", "
                                 << convertCmToMm(zpos) << ") with rotation: " << rot;
#endif
  }

  void constructInsideModule(cms::DDNamespace& ns, dd4hep::Volume& module, int mod) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: \t\tInside module ..." << mod;
#endif

    ///////////////////////////////////////////////////////////////
    //Pointers to the Rotation Matrices and to the Materials
    dd4hep::Rotation3D rot = getRotation(rotmat, ns);
    dd4hep::Material matter = ns.material(genMaterial);
    dd4hep::Material matplastic = ns.material(plastMat);

    double alpha = (1._pi) / nsectors;
    double zi = zminBlock[mod];

    for (int i = 0; i < layerN[mod]; i++) {
      std::string name;
      dd4hep::Solid solid;
      dd4hep::Volume glog, plog;
      int layer = getLayer(mod, i);
      double zo = zi + 0.5 * dzStep;

      for (int iphi = 0; iphi < phiSections; iphi++) {
        double ziAir = zo - thick[mod];
        double rinF, rinB;
        if (layer == 1) {
          rinF = ziAir * tan(angTop);
          rinB = zo * tan(angTop);
        } else {
          rinF = ziAir * tan(angBot);
          rinB = zo * tan(angBot);
        }
        double routF = (ziAir - z1Beam) * slope;
        double routB = (zo - z1Beam) * slope;
        if (routF > routBlock2[mod])
          routF = routBlock2[mod];
        if (routB > routBlock2[mod])
          routB = routBlock2[mod];

#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Layer " << i << " Phi " << iphi << " Front "
                                     << convertCmToMm(ziAir) << ", " << convertCmToMm(rinF) << ", "
                                     << convertCmToMm(routF) << " Back " << convertCmToMm(zo) << ", "
                                     << convertCmToMm(rinB) << ", " << convertCmToMm(routB);
#endif

        double yh1, bl1, tl1, yh2, bl2, tl2, theta, phi, alp;
        double xpos, ypos, zpos;
        parameterLayer(
            iphi, rinF, routF, rinB, routB, ziAir, zo, yh1, bl1, tl1, yh2, bl2, tl2, alp, theta, phi, xpos, ypos, zpos);

        name = DDSplit(module.name()).first + layerName[layer] + phiName[iphi] + "Air";
        solid = dd4hep::Trap(ns.prepend(name), 0.5 * thick[mod], theta, phi, yh1, bl1, tl1, alp, yh2, bl2, tl2, alp);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << solid.name() << " Trap made of " << matter.name()
                                     << " of dimensions " << convertCmToMm(0.5 * thick[mod]) << ", "
                                     << convertRadToDeg(theta) << ", " << convertRadToDeg(phi) << ", "
                                     << convertCmToMm(yh1) << ", " << convertCmToMm(bl1) << ", " << convertCmToMm(tl1)
                                     << ", " << convertRadToDeg(alp) << ", " << convertCmToMm(yh2) << ", "
                                     << convertCmToMm(bl2) << ", " << convertCmToMm(tl2) << ", "
                                     << convertRadToDeg(alp);
#endif

        glog = dd4hep::Volume(solid.name(), solid, matter);
        dd4hep::Position r1(xpos, ypos, zpos);
        module.placeVolume(glog, layer + 1, dd4hep::Transform3D(rot, r1));

#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << glog.name() << " number " << (layer + 1)
                                     << " positioned in " << module.name() << " at (" << convertCmToMm(xpos) << ", "
                                     << convertCmToMm(ypos) << ", " << convertCmToMm(zpos)
                                     << ") with rotation: " << rot;
#endif

        //Now the plastic with scintillators
        double yh = 0.5 * (routF - rinB) - getTrim(mod, iphi);
        double bl = 0.5 * rinB * tan(alpha) - getTrim(mod, iphi);
        double tl = 0.5 * routF * tan(alpha) - getTrim(mod, iphi);
        name = DDSplit(module.name()).first + layerName[layer] + phiName[iphi];
        solid = dd4hep::Trap(ns.prepend(name), 0.5 * layerT[layer], 0, 0, yh, bl, tl, alp, yh, bl, tl, alp);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << solid.name() << " Trap made of " << plastMat
                                     << " of dimensions " << convertCmToMm(0.5 * layerT[layer]) << ", 0, 0, "
                                     << convertCmToMm(yh) << ", " << convertCmToMm(bl) << ", " << convertCmToMm(tl)
                                     << ", " << convertRadToDeg(alp) << ", " << convertCmToMm(yh) << ", "
                                     << convertCmToMm(bl) << ", " << convertCmToMm(tl) << ", " << convertRadToDeg(alp);
#endif

        plog = dd4hep::Volume(solid.name(), solid, matplastic);
        ypos = 0.5 * (routF + rinB) - xpos;
        glog.placeVolume(plog, idOffset + layer + 1, dd4hep::Position(0., ypos, 0.));

#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << plog.name() << " number " << (idOffset + layer + 1)
                                     << " positioned in " << glog.name() << " at (0, " << convertCmToMm(ypos)
                                     << ", 0) with no rotation";
#endif

        //Constructing the scintillators inside
        int copyNo = layer * 10 + layerType[layer];
        name = modName[mod] + layerName[layer] + phiName[iphi];
        constructScintLayer(ns, plog, scintT[layer], yh, bl, tl, alp, name, copyNo);
        zo += 0.5 * dzStep;
      }  // End of loop over phi indices
      zi = zo - 0.5 * dzStep;
    }  // End of loop on layers
  }

  void constructScintLayer(cms::DDNamespace& ns,
                           dd4hep::Volume& detector,
                           double dz,
                           double yh,
                           double bl,
                           double tl,
                           double alp,
                           const std::string& nm,
                           int id) {
    dd4hep::Material matter = ns.material(scintMat);
    std::string name = idName + "Scintillator" + nm;

    dd4hep::Solid solid = dd4hep::Trap(ns.prepend(name), 0.5 * dz, 0, 0, yh, bl, tl, alp, yh, bl, tl, alp);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << solid.name() << " Trap made of " << scintMat
                                 << " of dimensions " << convertCmToMm(0.5 * dz) << ", 0, 0, " << convertCmToMm(yh)
                                 << ", " << convertCmToMm(bl) << ", " << convertCmToMm(tl) << ", "
                                 << convertRadToDeg(alp) << ", " << convertCmToMm(yh) << ", " << convertCmToMm(bl)
                                 << ", " << convertCmToMm(tl) << ", " << convertRadToDeg(alp);
#endif

    dd4hep::Volume glog(solid.name(), solid, matter);
    detector.placeVolume(glog, id);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << glog.name() << " number " << id << " positioned in "
                                 << detector.name() << " at (0,0,0) with no rotation";
#endif
  }

  int getLayer(unsigned int i, unsigned int j) const {
    switch (i) {
      case 0:
        return layerN0[j];
        break;
      case 1:
        return layerN1[j];
        break;
      case 2:
        return layerN2[j];
        break;
      case 3:
        return layerN3[j];
        break;
      case 4:
        return layerN4[j];
        break;
      case 5:
        return layerN5[j];
        break;
      default:
        return 0;
    }
  }

  double getTrim(unsigned int i, unsigned int j) const {
    if (j == 0)
      return trimLeft[i];
    else
      return trimRight[j];
  }

  void parameterLayer0(int mod,
                       int layer,
                       int iphi,
                       double& yh,
                       double& bl,
                       double& tl,
                       double& alp,
                       double& xpos,
                       double& ypos,
                       double& zpos) {
    //Given module and layer number compute parameters of trapezoid
    //and positioning parameters
    double alpha = (1._pi) / nsectors;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "Input " << iphi << " " << layer << " " << iphi << " Alpha "
                                 << convertRadToDeg(alpha);
#endif

    double zi, zo;
    if (iphi == 0) {
      zi = zminBlock[mod];
      zo = zi + layerT[layer];
    } else {
      zo = zmaxBlock[mod];
      zi = zo - layerT[layer];
    }
    double rin, rout;
    if (mod == 0) {
      rin = zo * tan(angTop);
      rout = (zi - z1Beam) * slope;
    } else {
      rin = zo * tan(angBot);
      rout = zi * tan(angTop);
    }
    yh = 0.5 * (rout - rin);
    bl = 0.5 * rin * tan(alpha);
    tl = 0.5 * rout * tan(alpha);
    xpos = 0.5 * (rin + rout);
    ypos = 0.5 * (bl + tl);
    zpos = 0.5 * (zi + zo);
    yh -= getTrim(mod, iphi);
    bl -= getTrim(mod, iphi);
    tl -= getTrim(mod, iphi);
    alp = atan(0.5 * tan(alpha));
    if (iphi == 0) {
      ypos = -ypos;
    } else {
      alp = -alp;
    }

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "Output Dimensions " << convertCmToMm(yh) << " " << convertCmToMm(bl) << " "
                                 << convertCmToMm(tl) << " " << convertRadToDeg(alp) << " Position "
                                 << convertCmToMm(xpos) << " " << convertCmToMm(ypos) << " " << convertCmToMm(zpos);
#endif
  }

  void parameterLayer(int iphi,
                      double rinF,
                      double routF,
                      double rinB,
                      double routB,
                      double zi,
                      double zo,
                      double& yh1,
                      double& bl1,
                      double& tl1,
                      double& yh2,
                      double& bl2,
                      double& tl2,
                      double& alp,
                      double& theta,
                      double& phi,
                      double& xpos,
                      double& ypos,
                      double& zpos) {
    //Given rin, rout compute parameters of the trapezoid and
    //position of the trapezoid for a standrd layer
    double alpha = (1._pi) / nsectors;

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "Input " << iphi << " Front " << convertCmToMm(rinF) << " " << convertCmToMm(routF)
                                 << " " << convertCmToMm(zi) << " Back " << convertCmToMm(rinB) << " "
                                 << convertCmToMm(routB) << " " << convertCmToMm(zo) << " Alpha "
                                 << convertRadToDeg(alpha);
#endif

    yh1 = 0.5 * (routF - rinB);
    bl1 = 0.5 * rinB * tan(alpha);
    tl1 = 0.5 * routF * tan(alpha);
    yh2 = 0.5 * (routF - rinB);
    bl2 = 0.5 * rinB * tan(alpha);
    tl2 = 0.5 * routF * tan(alpha);
    double dx = 0.25 * (bl2 + tl2 - bl1 - tl1);
    double dy = 0.5 * (rinB + routF - rinB - routF);
    xpos = 0.25 * (rinB + routF + rinB + routF);
    ypos = 0.25 * (bl2 + tl2 + bl1 + tl1);
    zpos = 0.5 * (zi + zo);
    alp = atan(0.5 * tan(alpha));
    //  ypos-= tolPos;
    if (iphi == 0) {
      ypos = -ypos;
    } else {
      alp = -alp;
      dx = -dx;
    }
    double r = sqrt(dx * dx + dy * dy);
    theta = atan(r / (zo - zi));
    phi = atan2(dy, dx);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "Output Dimensions " << convertCmToMm(yh1) << " " << convertCmToMm(bl1) << " "
                                 << convertCmToMm(tl1) << " " << convertCmToMm(yh2) << " " << convertCmToMm(bl2) << " "
                                 << convertCmToMm(tl2) << " " << convertRadToDeg(alp) << " " << convertRadToDeg(theta)
                                 << " " << convertRadToDeg(phi) << " Position " << convertCmToMm(xpos) << " "
                                 << convertCmToMm(ypos) << " " << convertCmToMm(zpos);
#endif
  }

  dd4hep::Rotation3D getRotation(const std::string& rotation, cms::DDNamespace& ns) {
    std::string rot = (strchr(rotation.c_str(), NAMESPACE_SEP) == nullptr) ? ("rotations:" + rotation) : rotation;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "getRotation: " << rotation << ":" << rot << ":" << ns.rotation(rot);
#endif
    return ns.rotation(rot);
  }
};

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  HCalEndcapAlgo hcalendcapalgo(ctxt, e);
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hcal_DDHCalEndcapAlgo, algorithm)
