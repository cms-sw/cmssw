///////////////////////////////////////////////////////////////////////////////
// File: DDHCalEndcapAlgo.cc
//   adapted from CCal(G4)HcalEndcap.cc
// Description: Geometry factory class for Hcal Endcap
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"

//#define EDM_ML_DEBUG
using namespace cms_units::operators;

class DDHCalEndcapAlgo : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDHCalEndcapAlgo();
  ~DDHCalEndcapAlgo() override;

  //Get Methods
  const std::string& getGenMat() const { return genMaterial; }
  const std::string& getRotation() const { return rotation; }
  int getNsectors() const { return nsectors; }
  int getNsectortot() const { return nsectortot; }
  int getEndcaps() const { return nEndcap; }
  int equipModule(unsigned int i) const { return eModule[i]; }
  double getZShift() const { return zShift; }

  double getZFront() const { return zFront; }
  double getZEnd() const { return zEnd; }
  double getZiNose() const { return ziNose; }
  double getZiL0Nose() const { return ziL0Nose; }
  double getZiBody() const { return ziBody; }
  double getZiL0Body() const { return ziL0Body; }
  double getZiKink() const { return ziKink; }
  double getZ0Beam() const { return z0Beam; }
  double getZ1Beam() const { return z1Beam; }
  double getZiDip() const { return ziDip; }
  double getDzStep() const { return dzStep; }
  double getDzShift() const { return dzShift; }
  double getZShiftHac2() const { return zShiftHac2; }

  double getRout() const { return rout; }
  double getRinKink() const { return riKink; }
  double getRinDip() const { return riDip; }
  double getRoutDip() const { return roDip; }
  double getHeboxDepth() const { return heboxDepth; }
  double getDrEnd() const { return drEnd; }
  double getAngTop() const { return angTop; }
  double getAngBot() const { return angBot; }
  double getAngGap() const { return angGap; }
  double getSlope() const { return slope; }

  const std::string& getAbsMat() const { return absMat; }
  int getModules() const { return modules; }
  const std::string& getModName(unsigned int i) const { return modName[i]; }
  const std::string& getModMat(unsigned int i) const { return modMat[i]; }
  int getModType(unsigned int i) const { return modType[i]; }
  int getSectionModule(unsigned i) const { return sectionModule[i]; }
  int getLayerN(unsigned int i) const { return layerN[i]; }
  int getLayer(unsigned int i, unsigned int j) const;
  double getThick(unsigned int i) const { return thick[i]; }
  double getTrim(unsigned int i, unsigned int j) const;
  double getZminBlock(unsigned i) const { return zminBlock[i]; }
  double getZmaxBlock(unsigned i) const { return zmaxBlock[i]; }
  double getRinBlock1(unsigned i) const { return rinBlock1[i]; }
  double getRinBlock2(unsigned i) const { return rinBlock2[i]; }
  double getRoutBlock1(unsigned i) const { return routBlock1[i]; }
  double getRoutBlock2(unsigned i) const { return routBlock2[i]; }

  int getPhi() const { return phiSections; }
  const std::string& getPhiName(unsigned int i) const { return phiName[i]; }
  int getLayers() const { return layers; }
  const std::string& getLayerName(unsigned int i) const { return layerName[i]; }
  int getLayerType(unsigned int i) const { return layerType[i]; }
  double getLayerT(unsigned int i) const { return layerT[i]; }
  double getScintT(unsigned int i) const { return scintT[i]; }
  const std::string& getPlastMat() const { return plastMat; }
  const std::string& getScintMat() const { return scintMat; }
  const std::string& getRotMat() const { return rotmat; }
  double getTolPos() const { return tolPos; }
  double getTolAbs() const { return tolAbs; }

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

protected:
  void constructGeneralVolume(DDCompactView& cpv);
  void constructInsideSector(const DDLogicalPart& sector, DDCompactView& cpv);
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
                      double& zcpv);
  void parameterLayer0(int mod,
                       int layer,
                       int iphi,
                       double& yh,
                       double& bl,
                       double& tl,
                       double& alp,
                       double& xpos,
                       double& ypos,
                       double& zcpv);
  void constructInsideModule0(const DDLogicalPart& module, int mod, DDCompactView& cpv);
  void constructInsideModule(const DDLogicalPart& module, int mod, DDCompactView& cpv);
  void constructScintLayer(const DDLogicalPart& glog,
                           double pDz,
                           double yh,
                           double bl,
                           double tl,
                           double alp,
                           const std::string& name,
                           int id,
                           DDCompactView& cpv);

private:
  std::string genMaterial;   //General material
  int nsectors;              //Number of potenital straight edges
  int nsectortot;            //Number of straight edges (actual)
  int nEndcap;               //Number of endcaps
  std::vector<int> eModule;  //Modules to be present in part i (?)
  std::string rotHalf;       //Rotation matrix for half
  std::string rotns;         //Name space for rotation
  std::string rotation;      //Rotation matrix to place in mother
  double zShift;             //needed for TB setup (move HE)

  double zFront;      //Z of the front section
  double zEnd;        //Outer Z of the HE
  double ziNose;      //Starting Z of the nose
  double ziL0Nose;    //Starting Z of layer 0 at nose
  double ziBody;      //Starting Z of the body
  double ziL0Body;    //Starting Z of layer 0 at body
  double ziKink;      //Position of the kink point
  double z0Beam;      //Position of gap front along z-axis
  double z1Beam;      //Position of gap end   along z-axis
  double ziDip;       //Starting Z of dipped part of body
  double dzStep;      //Width in Z of a layer
  double dzShift;     //Shift in Z for HE
  double zShiftHac2;  //needed for TB (remove part Hac2)

  double rout;        //Outer R of the HE
  double riKink;      //Inner radius at kink point
  double riDip;       //Inner radius at the dip point
  double roDip;       //Outer radius at the dip point
  double heboxDepth;  //Depth of the HE box
  double drEnd;       //Shift in R for the end absorber

  double angTop;  //Angle of top end of HE
  double angBot;  //Angle of the bottom end of HE
  double angGap;  //Gap angle (in degrees)
  double slope;   //Slope of the gap on HE side

  std::string absMat;                //Absorber     material
  int modules;                       //Number of modules
  std::vector<std::string> modName;  //Name
  std::vector<std::string> modMat;   //Material
  std::vector<int> modType;          //Type (0/1 for front/standard)
  std::vector<int> sectionModule;    //Number of sections in a module
  std::vector<int> layerN;           //Number of layers
  std::vector<int> layerN0;          //Layer numbers in section 0
  std::vector<int> layerN1;          //Layer numbers in section 1
  std::vector<int> layerN2;          //Layer numbers in section 2
  std::vector<int> layerN3;          //Layer numbers in section 3
  std::vector<int> layerN4;          //Layer numbers in section 4
  std::vector<int> layerN5;          //Layer numbers in section 5
  std::vector<double> thick;         //Thickness of absorber/air
  std::vector<double> trimLeft;      //Trimming of left  layers in module
  std::vector<double> trimRight;     //Trimming of right layers in module
  std::vector<double> zminBlock;     //Minimum Z
  std::vector<double> zmaxBlock;     //Maximum Z
  std::vector<double> rinBlock1;     //Inner Radius
  std::vector<double> routBlock1;    //Outer Radius at zmin
  std::vector<double> rinBlock2;     //Inner Radius
  std::vector<double> routBlock2;    //Outer Radius at zmax

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

  std::string idName;       //Name of the "parent" volume.
  std::string idNameSpace;  //Namespace of this and ALL sub-parts
  int idOffset;             // Geant4 ID's...    = 4000;

  double tolPos, tolAbs;  //Tolerances
};

DDHCalEndcapAlgo::DDHCalEndcapAlgo()
    : modMat(0),
      modType(0),
      sectionModule(0),
      layerN(0),
      layerN0(0),
      layerN1(0),
      layerN2(0),
      layerN3(0),
      layerN4(0),
      layerN5(0),
      thick(0),
      trimLeft(0),
      trimRight(0),
      zminBlock(0),
      zmaxBlock(0),
      rinBlock1(0),
      routBlock1(0),
      rinBlock2(0),
      routBlock2(0),
      layerType(0),
      layerT(0),
      scintT(0) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Creating an instance";
#endif
}

DDHCalEndcapAlgo::~DDHCalEndcapAlgo() {}

int DDHCalEndcapAlgo::getLayer(unsigned int i, unsigned int j) const {
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

double DDHCalEndcapAlgo::getTrim(unsigned int i, unsigned int j) const {
  if (j == 0)
    return trimLeft[i];
  else
    return trimRight[j];
}

void DDHCalEndcapAlgo::initialize(const DDNumericArguments& nArgs,
                                  const DDVectorArguments& vArgs,
                                  const DDMapArguments&,
                                  const DDStringArguments& sArgs,
                                  const DDStringVectorArguments& vsArgs) {
  genMaterial = sArgs["MaterialName"];
  rotation = sArgs["Rotation"];
  nsectors = int(nArgs["Sector"]);
  nsectortot = int(nArgs["SectorTot"]);
  nEndcap = int(nArgs["Endcap"]);
  rotHalf = sArgs["RotHalf"];
  rotns = sArgs["RotNameSpace"];
  zShift = nArgs["ZShift"];

  zFront = nArgs["ZFront"];
  zEnd = nArgs["ZEnd"];
  ziNose = nArgs["ZiNose"];
  ziL0Nose = nArgs["ZiL0Nose"];
  ziBody = nArgs["ZiBody"];
  ziL0Body = nArgs["ZiL0Body"];
  z0Beam = nArgs["Z0Beam"];
  ziDip = nArgs["ZiDip"];
  dzStep = nArgs["DzStep"];
  zShiftHac2 = nArgs["ZShiftHac2"];
  double gap = nArgs["Gap"];
  double z1 = nArgs["Z1"];
  double r1 = nArgs["R1"];
  rout = nArgs["Rout"];
  heboxDepth = nArgs["HEboxDepth"];
  drEnd = nArgs["DrEnd"];
  double etamin = nArgs["Etamin"];
  angBot = nArgs["AngBot"];
  angGap = nArgs["AngGap"];

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: General material " << genMaterial << "\tSectors " << nsectors
                               << ",  " << nsectortot << "\tEndcaps " << nEndcap << "\tRotation matrix for half "
                               << rotns << ":" << rotHalf << "\n\tzFront " << zFront << " zEnd " << zEnd << " ziNose "
                               << ziNose << " ziL0Nose " << ziL0Nose << " ziBody " << ziBody << " ziL0Body " << ziL0Body
                               << " z0Beam " << z0Beam << " ziDip " << ziDip << " dzStep " << dzStep << " Gap " << gap
                               << " z1 " << z1 << "\n\tr1 " << r1 << " rout " << rout << " HeboxDepth " << heboxDepth
                               << " drEnd " << drEnd << "\tetamin " << etamin << " Bottom angle " << angBot
                               << " Gap angle " << angGap << " Z-Shift " << zShift << " " << zShiftHac2;
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
                               << "\tDzShift " << dzShift << "\n\tz1Beam " << z1Beam << "\tziKink" << ziKink
                               << "\triKink " << riKink << "\triDip " << riDip << "\n\troDip " << roDip << "\tRotation "
                               << rotation;
#endif

  ///////////////////////////////////////////////////////////////
  //Modules
  absMat = sArgs["AbsMat"];
  modules = int(nArgs["Modules"]);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Number of modules " << modules << " and absorber material "
                               << absMat;
#endif

  modName = vsArgs["ModuleName"];
  modMat = vsArgs["ModuleMat"];
  modType = dbl_to_int(vArgs["ModuleType"]);
  sectionModule = dbl_to_int(vArgs["SectionModule"]);
  thick = vArgs["ModuleThick"];
  trimLeft = vArgs["TrimLeft"];
  trimRight = vArgs["TrimRight"];
  eModule = dbl_to_int(vArgs["EquipModule"]);
  layerN = dbl_to_int(vArgs["LayerN"]);
  layerN0 = dbl_to_int(vArgs["LayerN0"]);
  layerN1 = dbl_to_int(vArgs["LayerN1"]);
  layerN2 = dbl_to_int(vArgs["LayerN2"]);
  layerN3 = dbl_to_int(vArgs["LayerN3"]);
  layerN4 = dbl_to_int(vArgs["LayerN4"]);
  layerN5 = dbl_to_int(vArgs["LayerN5"]);

#ifdef EDM_ML_DEBUG
  for (int i = 0; i < modules; i++) {
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << modName[i] << " type " << modType[i] << " Sections "
                                 << sectionModule[i] << " thickness of absorber/air " << thick[i] << " trim "
                                 << trimLeft[i] << ", " << trimRight[i] << " equip module " << eModule[i] << " with "
                                 << layerN[i] << " layers";
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
  phiSections = int(nArgs["PhiSections"]);
  phiName = vsArgs["PhiName"];
  layers = int(nArgs["Layers"]);
  layerName = vsArgs["LayerName"];
  layerType = dbl_to_int(vArgs["LayerType"]);
  layerT = vArgs["LayerT"];
  scintT = vArgs["ScintT"];
  scintMat = sArgs["ScintMat"];
  plastMat = sArgs["PlastMat"];
  rotmat = sArgs["RotMat"];

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Phi Sections " << phiSections;
  for (int i = 0; i < phiSections; i++)
    edm::LogVerbatim("HCalGeom") << "\tName[" << i << "] : " << phiName[i];
  edm::LogVerbatim("HCalGeom") << "\tPlastic: " << plastMat << "\tScintillator: " << scintMat << "\tRotation matrix "
                               << rotns << ":" << rotmat << "\n\tNumber of layers " << layers;
  for (int i = 0; i < layers; i++) {
    edm::LogVerbatim("HCalGeom") << "\t" << layerName[i] << "\tType " << layerType[i] << "\tThickness " << layerT[i]
                                 << "\tScint.Thick " << scintT[i];
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
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Module " << i << "\tZ/Rin/Rout " << zminBlock[i] << ", "
                                 << zmaxBlock[i] << "/ " << rinBlock1[i] << ", " << rinBlock2[i] << "/ "
                                 << routBlock1[i] << ", " << routBlock2[i];
#endif

  idName = sArgs["MotherName"];
  idNameSpace = DDCurrentNamespace::ns();
  idOffset = int(nArgs["IdOffset"]);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Parent " << parent().name() << " idName " << idName
                               << " NameSpace " << idNameSpace << " Offset " << idOffset;
#endif

  tolPos = nArgs["TolPos"];
  tolAbs = nArgs["TolAbs"];

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Tolerances - Positioning " << tolPos << " Absorber " << tolAbs;
#endif
}

////////////////////////////////////////////////////////////////////
// DDHCalEndcapAlgo methods...
////////////////////////////////////////////////////////////////////

void DDHCalEndcapAlgo::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "==>> Constructing DDHCalEndcapAlgo...";
#endif

  constructGeneralVolume(cpv);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "<<== End of DDHCalEndcapAlgo construction ...";
#endif
}

//----------------------start here for DDD work!!! ---------------

void DDHCalEndcapAlgo::constructGeneralVolume(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: General volume...";
#endif

  bool proto = true;
  for (int i = 0; i < 3; i++)
    if (equipModule(i) > 0)
      proto = false;

  DDRotation rot;
  if (DDSplit(getRotation()).first == "NULL")
    rot = DDRotation();
  else
    rot = DDRotation(DDName(DDSplit(getRotation()).first, DDSplit(getRotation()).second));

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << " First " << DDSplit(getRotation()).first << " Second "
                               << DDSplit(getRotation()).second << " Rotation " << rot;
#endif

  DDTranslation r0(0, 0, getZShift());
  double alpha = (1._pi) / getNsectors();
  double dphi = getNsectortot() * (2._pi) / getNsectors();

  //!!!!!!!!!!!!!!!!!Should be zero. And removed as soon as
  //vertical walls are allowed in SolidPolyhedra
  double delz = 0;

  std::vector<double> pgonZ, pgonRmin, pgonRmax;
  if (proto) {
    double zf = getZiBody() + getZShiftHac2();
    pgonZ.emplace_back(zf - getDzShift());
    pgonRmin.emplace_back(zf * tan(getAngBot()));
    pgonRmax.emplace_back((zf - getZ1Beam()) * getSlope());
  } else {
    pgonZ.emplace_back(getZFront() - getDzShift());
    pgonRmin.emplace_back(getZFront() * tan(getAngTop()));
    pgonRmax.emplace_back((getZFront() - getZ1Beam()) * getSlope());
    pgonZ.emplace_back(getZiL0Body() - getDzShift());
    pgonRmin.emplace_back(getZiL0Body() * tan(getAngTop()));
    pgonRmax.emplace_back((getZiL0Body() - getZ1Beam()) * getSlope());
    pgonZ.emplace_back(getZiL0Body() - getDzShift());
    pgonRmin.emplace_back(getZiL0Body() * tan(getAngBot()));
    pgonRmax.emplace_back((getZiL0Body() - getZ1Beam()) * getSlope());
  }
  pgonZ.emplace_back(getZiKink() - getDzShift());
  pgonRmin.emplace_back(getRinKink());
  pgonRmax.emplace_back(getRout());
  pgonZ.emplace_back(getZiDip() - getDzShift());
  pgonRmin.emplace_back(getRinDip());
  pgonRmax.emplace_back(getRout());
  pgonZ.emplace_back(getZiDip() - getDzShift() + delz);
  pgonRmin.emplace_back(getRinDip());
  pgonRmax.emplace_back(getRoutDip());
  pgonZ.emplace_back(getZEnd() - getDzShift());
  pgonRmin.emplace_back(getZEnd() * tan(getAngBot()));
  pgonRmax.emplace_back(getRoutDip());
  pgonZ.emplace_back(getZEnd());
  pgonRmin.emplace_back(getZEnd() * tan(getAngBot()));
  pgonRmax.emplace_back(getRoutDip());

  std::string name("Null");
  DDSolid solid;
  solid =
      DDSolidFactory::polyhedra(DDName(idName, idNameSpace), getNsectortot(), -alpha, dphi, pgonZ, pgonRmin, pgonRmax);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << DDName(idName, idNameSpace) << " Polyhedra made of "
                               << getGenMat() << " with " << getNsectortot() << " sectors from "
                               << convertRadToDeg(-alpha) << " to " << convertRadToDeg(-alpha + dphi) << " and with "
                               << pgonZ.size() << " sections";
  for (unsigned int i = 0; i < pgonZ.size(); i++)
    edm::LogVerbatim("HCalGeom") << "\t\tZ = " << pgonZ[i] << "\tRmin = " << pgonRmin[i] << "\tRmax = " << pgonRmax[i];
#endif

  DDName matname(DDSplit(getGenMat()).first, DDSplit(getGenMat()).second);
  DDMaterial matter(matname);
  DDLogicalPart genlogic(DDName(idName, idNameSpace), matter, solid);

  DDName parentName = parent().name();
  cpv.position(DDName(idName, idNameSpace), parentName, 1, r0, rot);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << DDName(idName, idNameSpace) << " number 1 positioned in "
                               << parentName << " at " << r0 << " with " << rot;
#endif

  if (getEndcaps() != 1) {
    rot = DDRotation(DDName(rotHalf, rotns));
    cpv.position(DDName(idName, idNameSpace), parentName, 2, r0, rot);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << DDName(idName, idNameSpace) << " number 2 "
                                 << "positioned in " << parentName << " at " << r0 << " with " << rot;
#endif
  }

  //Forward half
  name = idName + "Front";
  std::vector<double> pgonZMod, pgonRminMod, pgonRmaxMod;
  for (unsigned int i = 0; i < (pgonZ.size() - 1); i++) {
    pgonZMod.emplace_back(pgonZ[i] + getDzShift());
    pgonRminMod.emplace_back(pgonRmin[i]);
    pgonRmaxMod.emplace_back(pgonRmax[i]);
  }
  solid = DDSolidFactory::polyhedra(
      DDName(name, idNameSpace), getNsectortot(), -alpha, dphi, pgonZMod, pgonRminMod, pgonRmaxMod);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << DDName(name, idNameSpace) << " Polyhedra made of "
                               << getGenMat() << " with " << getNsectortot() << " sectors from "
                               << convertRadToDeg(-alpha) << " to " << convertRadToDeg(-alpha + dphi) << " and with "
                               << pgonZMod.size() << " sections ";
  for (unsigned int i = 0; i < pgonZMod.size(); i++)
    edm::LogVerbatim("HCalGeom") << "\t\tZ = " << pgonZMod[i] << "\tRmin = " << pgonRminMod[i]
                                 << "\tRmax = " << pgonRmaxMod[i];
#endif

  DDLogicalPart genlogich(DDName(name, idNameSpace), matter, solid);

  cpv.position(genlogich, genlogic, 1, DDTranslation(0.0, 0.0, -getDzShift()), DDRotation());

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << genlogich.name() << " number 1 positioned in "
                               << genlogic.name() << " at (0,0," << -getDzShift() << ") with no rotation";
#endif

  //Construct sector (from -alpha to +alpha)
  name = idName + "Module";
  solid =
      DDSolidFactory::polyhedra(DDName(name, idNameSpace), 1, -alpha, 2 * alpha, pgonZMod, pgonRminMod, pgonRmaxMod);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << DDName(name, idNameSpace) << " Polyhedra made of "
                               << getGenMat() << " with 1 sector from " << convertRadToDeg(-alpha) << " to "
                               << convertRadToDeg(alpha) << " and with " << pgonZMod.size() << " sections";
  for (unsigned int i = 0; i < pgonZMod.size(); i++)
    edm::LogVerbatim("HCalGeom") << "\t\tZ = " << pgonZMod[i] << "\tRmin = " << pgonRminMod[i]
                                 << "\tRmax = " << pgonRmaxMod[i];
#endif

  DDLogicalPart seclogic(DDName(name, idNameSpace), matter, solid);

  for (int ii = 0; ii < getNsectortot(); ii++) {
    double phi = ii * 2 * alpha;
    DDRotation rotation;
    std::string rotstr("NULL");
    if (phi != 0) {
      rotstr = "R" + formatAsDegreesInInteger(phi);
      rotation = DDRotation(DDName(rotstr, rotns));
      if (!rotation) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Creating a new rotation " << rotstr << "\t 90,"
                                     << convertRadToDeg(phi) << ", 90," << convertRadToDeg(phi + 90._deg) << ", 0, 0";
#endif
        rotation = DDrot(DDName(rotstr, rotns), 90._deg, phi, 90._deg, (90._deg + phi), 0, 0);
      }  //if !rotation
    }    //if phi!=0

    cpv.position(seclogic, genlogich, ii + 1, DDTranslation(0.0, 0.0, 0.0), rotation);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << seclogic.name() << " number " << ii + 1 << " positioned in "
                                 << genlogich.name() << " at (0,0,0) with " << rotation;
#endif
  }

  //Construct the things inside the sector
  constructInsideSector(seclogic, cpv);

  //Backward half
  name = idName + "Back";
  std::vector<double> pgonZBack, pgonRminBack, pgonRmaxBack;
  pgonZBack.emplace_back(getZEnd() - getDzShift());
  pgonRminBack.emplace_back(pgonZBack[0] * tan(getAngBot()) + getDrEnd());
  pgonRmaxBack.emplace_back(getRoutDip());
  pgonZBack.emplace_back(getZEnd());
  pgonRminBack.emplace_back(pgonZBack[1] * tan(getAngBot()) + getDrEnd());
  pgonRmaxBack.emplace_back(getRoutDip());
  solid = DDSolidFactory::polyhedra(
      DDName(name, idNameSpace), getNsectortot(), -alpha, dphi, pgonZBack, pgonRminBack, pgonRmaxBack);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << DDName(name, idNameSpace) << " Polyhedra made of "
                               << getAbsMat() << " with " << getNsectortot() << " sectors from "
                               << convertRadToDeg(-alpha) << " to " << convertRadToDeg(-alpha + dphi) << " and with "
                               << pgonZBack.size() << " sections";
  for (unsigned int i = 0; i < pgonZBack.size(); i++)
    edm::LogVerbatim("HCalGeom") << "\t\tZ = " << pgonZBack[i] << "\tRmin = " << pgonRminBack[i]
                                 << "\tRmax = " << pgonRmaxBack[i];
#endif

  DDName absMatname(DDSplit(getAbsMat()).first, DDSplit(getAbsMat()).second);
  DDMaterial absMatter(absMatname);
  DDLogicalPart glog(DDName(name, idNameSpace), absMatter, solid);

  cpv.position(glog, genlogic, 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << glog.name() << " number 1 positioned in " << genlogic.name()
                               << " at (0,0,0) with no rotation";
#endif
}

void DDHCalEndcapAlgo::constructInsideSector(const DDLogicalPart& sector, DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Modules (" << getModules() << ") ...";
#endif

  double alpha = (1._pi) / getNsectors();

  for (int i = 0; i < getModules(); i++) {
    std::string name = idName + getModName(i);
    DDName matname(DDSplit(getModMat(i)).first, DDSplit(getModMat(i)).second);
    DDMaterial matter(matname);

    if (equipModule(i) > 0) {
      int nsec = getSectionModule(i);

      //!!!!!!!!!!!!!!!!!Should be zero. And removed as soon as
      //vertical walls are allowed in SolidPolyhedra
      double deltaz = 0;

      std::vector<double> pgonZ, pgonRmin, pgonRmax;
      if (nsec == 3) {
        double zf = getZminBlock(i) + getZShiftHac2();
        pgonZ.emplace_back(zf);
        pgonRmin.emplace_back(zf * tan(getAngBot()));
        pgonRmax.emplace_back((zf - getZ1Beam()) * getSlope());
        pgonZ.emplace_back(getZiKink());
        pgonRmin.emplace_back(getRinKink());
        pgonRmax.emplace_back(getRout());
      } else {
        pgonZ.emplace_back(getZminBlock(i));
        pgonRmin.emplace_back(getRinBlock1(i));
        pgonRmax.emplace_back(getRoutBlock1(i));
      }
      if (nsec == 4) {
        pgonZ.emplace_back(getZiDip());
        pgonRmin.emplace_back(getRinDip());
        pgonRmax.emplace_back(getRout());
        pgonZ.emplace_back(pgonZ[1] + deltaz);
        pgonRmin.emplace_back(pgonRmin[1]);
        pgonRmax.emplace_back(getRoutDip());
      }
      pgonZ.emplace_back(getZmaxBlock(i));
      pgonRmin.emplace_back(getRinBlock2(i));
      pgonRmax.emplace_back(getRoutBlock2(i));

      //Solid & volume
      DDSolid solid;
      solid = DDSolidFactory::polyhedra(DDName(name, idNameSpace), 1, -alpha, 2 * alpha, pgonZ, pgonRmin, pgonRmax);

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << DDName(name, idNameSpace) << " Polyhedra made of "
                                   << getModMat(i) << " with 1 sector from " << convertRadToDeg(-alpha) << " to "
                                   << convertRadToDeg(alpha) << " and with " << nsec << " sections";
      for (unsigned int k = 0; k < pgonZ.size(); k++)
        edm::LogVerbatim("HCalGeom") << "\t\tZ = " << pgonZ[k] << "\tRmin = " << pgonRmin[k]
                                     << "\tRmax = " << pgonRmax[k];
#endif

      DDLogicalPart glog(DDName(name, idNameSpace), matter, solid);

      cpv.position(glog, sector, i + 1, DDTranslation(0.0, 0.0, 0.0), DDRotation());

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << glog.name() << " number " << i + 1 << " positioned in "
                                   << sector.name() << " at (0,0,0) with no rotation";
#endif

      if (getModType(i) == 0)
        constructInsideModule0(glog, i, cpv);
      else
        constructInsideModule(glog, i, cpv);
    }
  }
}

void DDHCalEndcapAlgo::parameterLayer0(int mod,
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
  double alpha = (1._pi) / getNsectors();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Input " << iphi << " " << layer << " " << iphi << " Alpha "
                               << convertRadToDeg(alpha);
#endif

  double zi, zo;
  if (iphi == 0) {
    zi = getZminBlock(mod);
    zo = zi + getLayerT(layer);
  } else {
    zo = getZmaxBlock(mod);
    zi = zo - getLayerT(layer);
  }
  double rin, rout;
  if (mod == 0) {
    rin = zo * tan(getAngTop());
    rout = (zi - getZ1Beam()) * getSlope();
  } else {
    rin = zo * tan(getAngBot());
    rout = zi * tan(getAngTop());
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
  edm::LogVerbatim("HCalGeom") << "Output Dimensions " << yh << " " << bl << " " << tl << " " << convertRadToDeg(alp)
                               << " Position " << xpos << " " << ypos << " " << zpos;
#endif
}

void DDHCalEndcapAlgo::parameterLayer(int iphi,
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
  double alpha = (1._pi) / getNsectors();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Input " << iphi << " Front " << rinF << " " << routF << " " << zi << " Back " << rinB
                               << " " << routB << " " << zo << " Alpha " << convertRadToDeg(alpha);
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
  //  ypos-= getTolPos();
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
  edm::LogVerbatim("HCalGeom") << "Output Dimensions " << yh1 << " " << bl1 << " " << tl1 << " " << yh2 << " " << bl2
                               << " " << tl2 << " " << convertRadToDeg(alp) << " " << convertRadToDeg(theta) << " "
                               << convertRadToDeg(phi) << " Position " << xpos << " " << ypos << " " << zpos;
#endif
}

void DDHCalEndcapAlgo::constructInsideModule0(const DDLogicalPart& module, int mod, DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: \t\tInside module0 ..." << mod;
#endif

  ///////////////////////////////////////////////////////////////
  //Pointers to the Rotation Matrices and to the Materials
  std::string rotstr = getRotMat();
  DDRotation rot(DDName(rotstr, rotns));
  DDName matName(DDSplit(getAbsMat()).first, DDSplit(getAbsMat()).second);
  DDMaterial matabsorbr(matName);
  DDName plasName(DDSplit(getPlastMat()).first, DDSplit(getPlastMat()).second);
  DDMaterial matplastic(plasName);

  int layer = getLayer(mod, 0);
  int layer0 = getLayer(mod, 1);
  std::string name;
  double xpos, ypos, zpos;
  DDSolid solid;
  DDLogicalPart glog, plog;
  for (int iphi = 0; iphi < getPhi(); iphi++) {
    double yh, bl, tl, alp;
    parameterLayer0(mod, layer, iphi, yh, bl, tl, alp, xpos, ypos, zpos);
    name = module.name().name() + getLayerName(layer) + getPhiName(iphi);
    solid =
        DDSolidFactory::trap(DDName(name, idNameSpace), 0.5 * getLayerT(layer), 0, 0, yh, bl, tl, alp, yh, bl, tl, alp);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << solid.name() << " Trap made of " << getPlastMat()
                                 << " of dimensions " << 0.5 * getLayerT(layer) << ", 0, 0, " << yh << ", " << bl
                                 << ", " << tl << ", " << convertRadToDeg(alp) << ", " << yh << ", " << bl << ", " << tl
                                 << ", " << convertRadToDeg(alp);
#endif

    glog = DDLogicalPart(solid.ddname(), matplastic, solid);

    DDTranslation r1(xpos, ypos, zpos);
    cpv.position(glog, module, idOffset + layer + 1, r1, rot);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << glog.name() << " number " << idOffset + layer + 1
                                 << " positioned in " << module.name() << " at " << r1 << " with " << rot;
#endif

    //Now construct the layer of scintillator inside this
    int copyNo = layer0 * 10 + getLayerType(layer);
    name = getModName(mod) + getLayerName(layer) + getPhiName(iphi);
    constructScintLayer(glog, getScintT(layer), yh, bl, tl, alp, name, copyNo, cpv);
  }

  //Now the absorber layer
  double zi = getZminBlock(mod) + getLayerT(layer);
  double zo = zi + 0.5 * getDzStep();
  double rinF, routF, rinB, routB;
  if (mod == 0) {
    rinF = zi * tan(getAngTop());
    routF = (zi - getZ1Beam()) * getSlope();
    rinB = zo * tan(getAngTop());
    routB = (zo - getZ1Beam()) * getSlope();
  } else {
    rinF = zi * tan(getAngBot());
    routF = zi * tan(getAngTop());
    rinB = zo * tan(getAngBot());
    routB = zo * tan(getAngTop());
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Module " << mod << " Front " << zi << ", " << rinF << ", " << routF
                               << " Back " << zo << ", " << rinB << ", " << routB;
#endif

  double yh1, bl1, tl1, yh2, bl2, tl2, theta, phi, alp;
  parameterLayer(0, rinF, routF, rinB, routB, zi, zo, yh1, bl1, tl1, yh2, bl2, tl2, alp, theta, phi, xpos, ypos, zpos);
  double fact = getTolAbs();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Trim " << fact << " Param " << yh1 << ", " << bl1 << ", " << tl1
                               << ", " << yh2 << ", " << bl2 << ", " << tl2;
#endif

  bl1 -= fact;
  tl1 -= fact;
  bl2 -= fact;
  tl2 -= fact;

  name = module.name().name() + "Absorber";
  solid = DDSolidFactory::trap(
      DDName(name, idNameSpace), 0.5 * getThick(mod), theta, phi, yh1, bl1, tl1, alp, yh2, bl2, tl2, alp);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << solid.name() << " Trap made of " << getAbsMat()
                               << " of dimensions " << 0.5 * getThick(mod) << ", " << convertRadToDeg(theta) << ", "
                               << convertRadToDeg(phi) << ", " << yh1 << ", " << bl1 << ", " << tl1 << ", "
                               << convertRadToDeg(alp) << ", " << yh2 << ", " << bl2 << ", " << tl2 << ", "
                               << convertRadToDeg(alp);
#endif

  glog = DDLogicalPart(solid.ddname(), matabsorbr, solid);

  DDTranslation r2(xpos, ypos, zpos);
  cpv.position(glog, module, 1, r2, rot);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << glog.name() << " number 1 positioned in " << module.name()
                               << " at " << r2 << " with " << rot;
#endif
}

void DDHCalEndcapAlgo::constructInsideModule(const DDLogicalPart& module, int mod, DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: \t\tInside module ..." << mod;
#endif

  ///////////////////////////////////////////////////////////////
  //Pointers to the Rotation Matrices and to the Materials
  std::string rotstr = getRotMat();
  DDRotation rot(DDName(rotstr, rotns));
  DDName matName(DDSplit(getGenMat()).first, DDSplit(getGenMat()).second);
  DDMaterial matter(matName);
  DDName plasName(DDSplit(getPlastMat()).first, DDSplit(getPlastMat()).second);
  DDMaterial matplastic(plasName);

  double alpha = (1._pi) / getNsectors();
  double zi = getZminBlock(mod);

  for (int i = 0; i < getLayerN(mod); i++) {
    std::string name;
    DDSolid solid;
    DDLogicalPart glog, plog;
    int layer = getLayer(mod, i);
    double zo = zi + 0.5 * getDzStep();

    for (int iphi = 0; iphi < getPhi(); iphi++) {
      double ziAir = zo - getThick(mod);
      double rinF, rinB;
      if (layer == 1) {
        rinF = ziAir * tan(getAngTop());
        rinB = zo * tan(getAngTop());
      } else {
        rinF = ziAir * tan(getAngBot());
        rinB = zo * tan(getAngBot());
      }
      double routF = (ziAir - getZ1Beam()) * getSlope();
      double routB = (zo - getZ1Beam()) * getSlope();
      if (routF > getRoutBlock2(mod))
        routF = getRoutBlock2(mod);
      if (routB > getRoutBlock2(mod))
        routB = getRoutBlock2(mod);

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: Layer " << i << " Phi " << iphi << " Front " << ziAir << ", "
                                   << rinF << ", " << routF << " Back " << zo << ", " << rinB << ", " << routB;
#endif

      double yh1, bl1, tl1, yh2, bl2, tl2, theta, phi, alp;
      double xpos, ypos, zpos;
      parameterLayer(
          iphi, rinF, routF, rinB, routB, ziAir, zo, yh1, bl1, tl1, yh2, bl2, tl2, alp, theta, phi, xpos, ypos, zpos);

      name = module.name().name() + getLayerName(layer) + getPhiName(iphi) + "Air";
      solid = DDSolidFactory::trap(
          DDName(name, idNameSpace), 0.5 * getThick(mod), theta, phi, yh1, bl1, tl1, alp, yh2, bl2, tl2, alp);

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << solid.name() << " Trap made of " << getGenMat()
                                   << " of dimensions " << 0.5 * getThick(mod) << ", " << convertRadToDeg(theta) << ", "
                                   << convertRadToDeg(phi) << ", " << yh1 << ", " << bl1 << ", " << tl1 << ", "
                                   << convertRadToDeg(alp) << ", " << yh2 << ", " << bl2 << ", " << tl2 << ", "
                                   << convertRadToDeg(alp);
#endif

      glog = DDLogicalPart(solid.ddname(), matter, solid);

      DDTranslation r1(xpos, ypos, zpos);
      cpv.position(glog, module, layer + 1, r1, rot);

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << glog.name() << " number " << layer + 1
                                   << " positioned in " << module.name() << " at " << r1 << " with " << rot;
#endif

      //Now the plastic with scintillators
      double yh = 0.5 * (routF - rinB) - getTrim(mod, iphi);
      double bl = 0.5 * rinB * tan(alpha) - getTrim(mod, iphi);
      double tl = 0.5 * routF * tan(alpha) - getTrim(mod, iphi);
      name = module.name().name() + getLayerName(layer) + getPhiName(iphi);
      solid = DDSolidFactory::trap(
          DDName(name, idNameSpace), 0.5 * getLayerT(layer), 0, 0, yh, bl, tl, alp, yh, bl, tl, alp);

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << solid.name() << " Trap made of " << getPlastMat()
                                   << " of dimensions " << 0.5 * getLayerT(layer) << ", 0, 0, " << yh << ", " << bl
                                   << ", " << tl << ", " << convertRadToDeg(alp) << ", " << yh << ", " << bl << ", "
                                   << tl << ", " << convertRadToDeg(alp);
#endif

      plog = DDLogicalPart(solid.ddname(), matplastic, solid);

      ypos = 0.5 * (routF + rinB) - xpos;
      DDTranslation r2(0., ypos, 0.);
      cpv.position(plog, glog, idOffset + layer + 1, r2, DDRotation());

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << plog.name() << " number " << idOffset + layer + 1
                                   << " positioned in " << glog.name() << " at " << r2 << " with no rotation";
#endif

      //Constructin the scintillators inside
      int copyNo = layer * 10 + getLayerType(layer);
      name = getModName(mod) + getLayerName(layer) + getPhiName(iphi);
      constructScintLayer(plog, getScintT(layer), yh, bl, tl, alp, name, copyNo, cpv);
      zo += 0.5 * getDzStep();
    }  // End of loop over phi indices
    zi = zo - 0.5 * getDzStep();
  }  // End of loop on layers
}

void DDHCalEndcapAlgo::constructScintLayer(const DDLogicalPart& detector,
                                           double dz,
                                           double yh,
                                           double bl,
                                           double tl,
                                           double alp,
                                           const std::string& nm,
                                           int id,
                                           DDCompactView& cpv) {
  DDName matname(DDSplit(getScintMat()).first, DDSplit(getScintMat()).second);
  DDMaterial matter(matname);
  std::string name = idName + "Scintillator" + nm;

  DDSolid solid = DDSolidFactory::trap(DDName(name, idNameSpace), 0.5 * dz, 0, 0, yh, bl, tl, alp, yh, bl, tl, alp);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << DDName(name, idNameSpace) << " Trap made of " << getScintMat()
                               << " of dimensions " << 0.5 * dz << ", 0, 0, " << yh << ", " << bl << ", " << tl << ", "
                               << convertRadToDeg(alp) << ", " << yh << ", " << bl << ", " << tl << ", "
                               << convertRadToDeg(alp);
#endif

  DDLogicalPart glog(solid.ddname(), matter, solid);

  cpv.position(glog, detector, id, DDTranslation(0, 0, 0), DDRotation());

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapAlgo: " << glog.name() << " number " << id << " positioned in "
                               << detector.name() << " at (0,0,0) with no rotation";
#endif
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHCalEndcapAlgo, "hcal:DDHCalEndcapAlgo");
