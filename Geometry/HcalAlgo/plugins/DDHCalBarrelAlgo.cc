/////////////////////////////////////////////////////////////////////////////
// File: DDHCalBarrelAlgo.cc
//   adapted from CCal(G4)HcalBarrel.cc
// Description: Geometry factory class for Hcal Barrel
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "DataFormats/Math/interface/angle_units.h"
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
using namespace angle_units::operators;

class DDHCalBarrelAlgo : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDHCalBarrelAlgo();
  ~DDHCalBarrelAlgo() override;

  //Get Methods
  const std::string& getGenMaterial() const { return genMaterial; }
  int getNsectors() const { return nsectors; }
  int getNsectortot() const { return nsectortot; }
  int getNhalf() const { return nhalf; }
  double getRin() const { return rin; }
  double getRout() const { return rout; }
  int getRzones() const { return rzones; }
  double getTanTheta(unsigned int i) const { return ttheta[i]; }
  double getTheta(unsigned int i) const { return theta[i]; }
  double getRmax(unsigned int i) const { return rmax[i]; }
  double getZoff(unsigned int i) const { return zoff[i]; }

  int getNLayers() const { return nLayers; }
  int getLayerId(unsigned i) const { return layerId[i]; }
  const std::string& getLayerLabel(unsigned i) const { return layerLabel[i]; }
  const std::string& getLayerMaterial(unsigned i) const { return layerMat[i]; }
  double getLayerWidth(unsigned i) const { return layerWidth[i]; }
  double getLayerD1(unsigned i) const { return layerD1[i]; }
  double getLayerD2(unsigned i) const { return layerD2[i]; }
  double getLayerAlpha(unsigned i) const { return layerAlpha[i]; }
  double getLayerT1(unsigned i) const { return layerT1[i]; }
  double getLayerT2(unsigned i) const { return layerT2[i]; }
  int getLayerAbsorb(unsigned int i) const { return layerAbsorb[i]; }
  double getLayerGap(unsigned int i) const { return layerGap[i]; }

  const std::string& getSideMat(unsigned int i) const { return sideMat[i]; }
  double getSideD(unsigned int i) const { return sideD[i]; }
  double getSideT(unsigned int i) const { return sideT[i]; }
  int getSideAbsorber() const { return nSideAbs; }
  const std::string& getSideAbsName(unsigned int i) const { return sideAbsName[i]; }
  const std::string& getSideAbsMat(unsigned int i) const { return sideAbsMat[i]; }
  double getSideAbsW(unsigned int i) const { return sideAbsW[i]; }

  int getAbsorberN() const { return nAbsorber; }
  const std::string& getAbsorbName(unsigned int i) const { return absorbName[i]; }
  const std::string& getAbsorbMat(unsigned int i) const { return absorbMat[i]; }
  double getAbsorbD(unsigned int i) const { return absorbD[i]; }
  double getAbsorbT(unsigned int i) const { return absorbT[i]; }
  const std::string& getMiddleMat() const { return middleMat; }
  double getMiddleD() const { return middleD; }
  double getMiddleW() const { return middleW; }
  int getMidAbsorber() const { return nMidAbs; }
  const std::string& getMidAbsName(unsigned int i) const { return midName[i]; }
  const std::string& getMidAbsMat(unsigned int i) const { return midMat[i]; }
  double getMidAbsW(unsigned int i) const { return midW[i]; }
  double getMidAbsT(unsigned int i) const { return midT[i]; }

  const std::string& getDetMat() const { return detMat; }
  const std::string& getDetMatPl() const { return detMatPl; }
  const std::string& getDetMatSc() const { return detMatSc; }
  int getDetType(unsigned int i) const { return detType[i]; }
  double getDetdP1(unsigned int i) const { return detdP1[i]; }
  double getDetdP2(unsigned int i) const { return detdP2[i]; }
  double getDetT11(unsigned int i) const { return detT11[i]; }
  double getDetT12(unsigned int i) const { return detT12[i]; }
  double getDetTsc(unsigned int i) const { return detTsc[i]; }
  double getDetT21(unsigned int i) const { return detT21[i]; }
  double getDetT22(unsigned int i) const { return detT22[i]; }
  double getDetWidth1(unsigned int i) const { return detWidth1[i]; }
  double getDetWidth2(unsigned int i) const { return detWidth2[i]; }
  int getDetPosY(unsigned int i) const { return detPosY[i]; }

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  void constructGeneralVolume(DDCompactView& cpv);
  void constructInsideSector(const DDLogicalPart& sector, DDCompactView& cpv);
  void constructInsideLayers(const DDLogicalPart& laylog,
                             const std::string& name,
                             int id,
                             int nAbs,
                             double rin,
                             double d1,
                             double alpha1,
                             double d2,
                             double alpha2,
                             double t1,
                             double t2,
                             DDCompactView& cpv);
  DDLogicalPart constructSideLayer(
      const DDLogicalPart& laylog, const std::string& nm, int nAbs, double rin, double alpha, DDCompactView& cpv);
  DDLogicalPart constructMidLayer(
      const DDLogicalPart& laylog, const std::string& nm, double rin, double alpha, DDCompactView& cpv);
  void constructInsideDetectors(const DDLogicalPart& detector,
                                const std::string& name,
                                int id,
                                double dx,
                                double dy,
                                double dz,
                                int type,
                                DDCompactView& cpv);

  //General Volume
  //      <----- Zmax ------>
  // Rout ************************-------
  //      *                      *Rstep2|        Theta angle w.r.t. vertical
  //      *                      *---------------
  //      *                     *               |
  //      *                    *Theta[i]        Rmax[i]
  //      *                   *---------------  |
  //                        *Theta[0] Rmax[0]|  |
  // Rin  *****************----------------------

  std::string genMaterial;     //General material
  int nsectors;                //Number of potenital straight edges
  int nsectortot;              //Number of straight edges (actual)
  int nhalf;                   //Number of half modules
  double rin, rout;            //See picture
  int rzones;                  //  ....
  std::vector<double> theta;   //  .... (in degrees)
  std::vector<double> rmax;    //  ....
  std::vector<double> zoff;    //  ....
  std::vector<double> ttheta;  //tan(theta)
  std::string rotHalf;         //Rotation matrix of the second half
  std::string rotns;           //Name space for Rotation matrices

  //Upper layers inside general volume
  //     <---- Zout ---->
  //  |  ****************     |
  //  |  *              *     Wstep
  //  W  *              ***** |
  //  |  *                  *
  //  |  ********************
  //     <------ Zin ------->
  // Middle layers inside general volume
  //     <------ Zout ------>         Zout = Full sector Z at position
  //  |  ********************         Zin  = Full sector Z at position
  //  |  *                 *
  //  W  *                * Angle = Theta sector
  //  |  *               *  )
  //  |  ****************--------
  //     <------ Zin ------->

  // Lower layers
  //     <------ Zout ------>         Zin(i)=Zout(i-1)
  //  |  ********************         Zout(i)=Zin(i)+W(i)/tan(Theta(i))
  //  |  *                 *
  //  W  *                *  Theta
  //  |  *               *
  //  |  ****************--------
  //     <--- Zin ------>

  int nLayers;                          //Number of layers
  std::vector<int> layerId;             //Number identification
  std::vector<std::string> layerLabel;  //String identification
  std::vector<std::string> layerMat;    //Material
  std::vector<double> layerWidth;       //W in picture
  std::vector<double> layerD1;          //d1 in front picture
  std::vector<double> layerD2;          //d2 in front picture
  std::vector<double> layerAlpha;       //Angular width of the middle tiles
  std::vector<double> layerT1;          //t in front picture (side)
  std::vector<double> layerT2;          //t in front picture (front)
  std::vector<int> layerAbsorb;         //Absorber flag
  std::vector<double> layerGap;         //Gap at the edge

  int nAbsorber;                        //Number of absorber layers in middle
  std::vector<std::string> absorbName;  //Absorber name
  std::vector<std::string> absorbMat;   //Absorber material
  std::vector<double> absorbD;          //Distance from the bottom surface
  std::vector<double> absorbT;          //Thickness
  std::string middleMat;                //Material of the detector layer
  double middleD;                       //Distance from the bottom surface
  double middleW;                       //Half width
  int nMidAbs;                          //Number of absorbers in front part
  std::vector<std::string> midName;     //Absorber names in the front part
  std::vector<std::string> midMat;      //Absorber material
  std::vector<double> midW;             //Half width
  std::vector<double> midT;             //Thickness

  std::vector<std::string> sideMat;      //Material for special side layers
  std::vector<double> sideD;             //Depth from bottom surface
  std::vector<double> sideT;             //Thickness
  int nSideAbs;                          //Number of absorbers in special side
  std::vector<std::string> sideAbsName;  //Absorber name
  std::vector<std::string> sideAbsMat;   //Absorber material
  std::vector<double> sideAbsW;          //Half width

  // Detectors. Each volume inside the layer has the shape:
  //
  // ******************************* |
  // *\\\\\\\Plastic\\\\\\\\\\\\\\\* T2
  // ******************************* |
  // *////Scintillator/////////////* Tsc
  // ******************************* |
  // *\\\\\\\Plastic\\\\\\\\\\\\\\\* T1
  // ******************************* |   |
  // *         Air                 *     dP1
  // *******************************     |
  //
  std::string detMat;    //fill material
  std::string detRot;    //Rotation matrix for the 2nd
  std::string detMatPl;  //Plastic material
  std::string detMatSc;  //Scintillator material
  std::vector<int> detType;
  std::vector<double> detdP1;     //Air gap (side)
  std::vector<double> detdP2;     //Air gap (centre)
  std::vector<double> detT11;     //Back plastic thickness (side)
  std::vector<double> detT12;     //Back plastic thickness (centre)
  std::vector<double> detTsc;     //Scintillator
  std::vector<double> detT21;     //Front plastic thickness (side)
  std::vector<double> detT22;     //Front plastic thickness (centre)
  std::vector<double> detWidth1;  //Width of phi(1,4) megatiles
  std::vector<double> detWidth2;  //Width of phi(2,3) megatiles
  std::vector<int> detPosY;       //Positioning of phi(1,4) tiles - 0 centre

  std::string idName;       //Name of the "parent" volume.
  std::string idNameSpace;  //Namespace of this and ALL sub-parts
  int idOffset;             // Geant4 ID's...    = 3000;
};

DDHCalBarrelAlgo::DDHCalBarrelAlgo()
    : theta(0),
      rmax(0),
      zoff(0),
      ttheta(0),
      layerId(0),
      layerLabel(0),
      layerMat(0),
      layerWidth(0),
      layerD1(0),
      layerD2(0),
      layerAlpha(0),
      layerT1(0),
      layerT2(0),
      layerAbsorb(0),
      layerGap(0),
      absorbName(0),
      absorbMat(0),
      absorbD(0),
      absorbT(0),
      midName(0),
      midMat(0),
      midW(0),
      midT(0),
      sideMat(0),
      sideD(0),
      sideT(0),
      sideAbsName(0),
      sideAbsMat(0),
      sideAbsW(0),
      detType(0),
      detdP1(0),
      detdP2(0),
      detT11(0),
      detT12(0),
      detTsc(0),
      detT21(0),
      detT22(0),
      detWidth1(0),
      detWidth2(0),
      detPosY(0) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Creating an instance";
#endif
}

DDHCalBarrelAlgo::~DDHCalBarrelAlgo() {}

void DDHCalBarrelAlgo::initialize(const DDNumericArguments& nArgs,
                                  const DDVectorArguments& vArgs,
                                  const DDMapArguments&,
                                  const DDStringArguments& sArgs,
                                  const DDStringVectorArguments& vsArgs) {
  genMaterial = sArgs["MaterialName"];
  nsectors = int(nArgs["NSector"]);
  nsectortot = int(nArgs["NSectorTot"]);
  nhalf = int(nArgs["NHalf"]);
  rin = nArgs["RIn"];
  rout = nArgs["ROut"];
  rzones = int(nArgs["RZones"]);
  rotHalf = sArgs["RotHalf"];
  rotns = sArgs["RotNameSpace"];

  theta = vArgs["Theta"];
  rmax = vArgs["RMax"];
  zoff = vArgs["ZOff"];
  for (int i = 0; i < rzones; i++) {
    ttheta.emplace_back(tan(theta[i]));  //*deg already done in XML
  }
  if (rzones > 3)
    rmax[2] = (zoff[3] - zoff[2]) / ttheta[2];

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: General material " << genMaterial << "\tSectors " << nsectors
                               << ", " << nsectortot << "\tHalves " << nhalf << "\tRotation matrix " << rotns << ":"
                               << rotHalf << "\n\t\t" << rin << "\t" << rout << "\t" << rzones;
  for (int i = 0; i < rzones; i++)
    edm::LogVerbatim("HCalGeom") << "\tTheta[" << i << "] = " << theta[i] << "\trmax[" << i << "] = " << rmax[i]
                                 << "\tzoff[" << i << "] = " << zoff[i];
#endif
  ///////////////////////////////////////////////////////////////
  //Layers
  nLayers = int(nArgs["NLayers"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Layer\t" << nLayers;
#endif
  layerId = dbl_to_int(vArgs["Id"]);
  layerLabel = vsArgs["LayerLabel"];
  layerMat = vsArgs["LayerMat"];
  layerWidth = vArgs["LayerWidth"];
  layerD1 = vArgs["D1"];
  layerD2 = vArgs["D2"];
  layerAlpha = vArgs["Alpha2"];
  layerT1 = vArgs["T1"];
  layerT2 = vArgs["T2"];
  layerAbsorb = dbl_to_int(vArgs["AbsL"]);
  layerGap = vArgs["Gap"];
#ifdef EDM_ML_DEBUG
  for (int i = 0; i < nLayers; i++)
    edm::LogVerbatim("HCalGeom") << layerLabel[i] << "\t" << layerId[i] << "\t" << layerMat[i] << "\t" << layerWidth[i]
                                 << "\t" << layerD1[i] << "\t" << layerD2[i] << "\t" << layerAlpha[i] << "\t"
                                 << layerT1[i] << "\t" << layerT2[i] << "\t" << layerAbsorb[i] << "\t" << layerGap[i];
#endif

  ///////////////////////////////////////////////////////////////
  //Absorber Layers and middle part
  absorbName = vsArgs["AbsorbName"];
  absorbMat = vsArgs["AbsorbMat"];
  absorbD = vArgs["AbsorbD"];
  absorbT = vArgs["AbsorbT"];
  nAbsorber = absorbName.size();
#ifdef EDM_ML_DEBUG
  for (int i = 0; i < nAbsorber; i++)
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << absorbName[i] << " Material " << absorbMat[i] << " d "
                                 << absorbD[i] << " t " << absorbT[i];
#endif
  middleMat = sArgs["MiddleMat"];
  middleD = nArgs["MiddleD"];
  middleW = nArgs["MiddleW"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Middle material " << middleMat << " d " << middleD << " w "
                               << middleW;
#endif
  midName = vsArgs["MidAbsName"];
  midMat = vsArgs["MidAbsMat"];
  midW = vArgs["MidAbsW"];
  midT = vArgs["MidAbsT"];
  nMidAbs = midName.size();
#ifdef EDM_ML_DEBUG
  for (int i = 0; i < nMidAbs; i++)
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << midName[i] << " Material " << midMat[i] << " W " << midW[i]
                                 << " T " << midT[i];
#endif

  //Absorber layers in the side part
  sideMat = vsArgs["SideMat"];
  sideD = vArgs["SideD"];
  sideT = vArgs["SideT"];
#ifdef EDM_ML_DEBUG
  for (unsigned int i = 0; i < sideMat.size(); i++)
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Side material " << sideMat[i] << " d " << sideD[i] << " t "
                                 << sideT[i];
#endif
  sideAbsName = vsArgs["SideAbsName"];
  sideAbsMat = vsArgs["SideAbsMat"];
  sideAbsW = vArgs["SideAbsW"];
  nSideAbs = sideAbsName.size();
#ifdef EDM_ML_DEBUG
  for (int i = 0; i < nSideAbs; i++)
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << sideAbsName[i] << " Material " << sideAbsMat[i] << " W "
                                 << sideAbsW[i];
#endif

  ///////////////////////////////////////////////////////////////
  // Detectors

  detMat = sArgs["DetMat"];
  detRot = sArgs["DetRot"];
  detMatPl = sArgs["DetMatPl"];
  detMatSc = sArgs["DetMatSc"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Detector (" << nLayers << ") Rotation matrix " << rotns << ":"
                               << detRot << "\n\t\t" << detMat << "\t" << detMatPl << "\t" << detMatSc;
#endif
  detType = dbl_to_int(vArgs["DetType"]);
  detdP1 = vArgs["DetdP1"];
  detdP2 = vArgs["DetdP2"];
  detT11 = vArgs["DetT11"];
  detT12 = vArgs["DetT12"];
  detTsc = vArgs["DetTsc"];
  detT21 = vArgs["DetT21"];
  detT22 = vArgs["DetT22"];
  detWidth1 = vArgs["DetWidth1"];
  detWidth2 = vArgs["DetWidth2"];
  detPosY = dbl_to_int(vArgs["DetPosY"]);
#ifdef EDM_ML_DEBUG
  for (int i = 0; i < nLayers; i++)
    edm::LogVerbatim("HCalGeom") << i + 1 << "\t" << detType[i] << "\t" << detdP1[i] << ", " << detdP2[i] << "\t"
                                 << detT11[i] << ", " << detT12[i] << "\t" << detTsc[i] << "\t" << detT21[i] << ", "
                                 << detT22[i] << "\t" << detWidth1[i] << "\t" << detWidth2[i] << "\t" << detPosY[i];
#endif

  //  idName = parentName.name();
  idName = sArgs["MotherName"];
  idNameSpace = DDCurrentNamespace::ns();
  idOffset = int(nArgs["IdOffset"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Parent " << parent().name() << " idName " << idName
                               << " NameSpace " << idNameSpace << " Offset " << idOffset;
#endif
}

////////////////////////////////////////////////////////////////////
// DDHCalBarrelAlgo methods...
////////////////////////////////////////////////////////////////////

void DDHCalBarrelAlgo::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "==>> Constructing DDHCalBarrelAlgo...";
#endif
  constructGeneralVolume(cpv);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "<<== End of DDHCalBarrelAlgo construction";
#endif
}

//----------------------start here for DDD work!!! ---------------

void DDHCalBarrelAlgo::constructGeneralVolume(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: General volume...";
#endif

  DDRotation rot = DDRotation();

  double alpha = (1._pi) / getNsectors();
  double dphi = getNsectortot() * (2._pi) / getNsectors();
  int nsec, ntot = 15;
  if (getNhalf() == 1)
    nsec = 8;
  else
    nsec = 15;
  int nf = ntot - nsec;

  //Calculate zmin... see HCalBarrel.hh picture. For polyhedra
  //Rmin and Rmax are distances to vertex
  double zmax = getZoff(3);
  double zstep5 = getZoff(4);
  double zstep4 = (getZoff(1) + getRmax(1) * getTanTheta(1));
  if ((getZoff(2) + getRmax(1) * getTanTheta(2)) > zstep4)
    zstep4 = (getZoff(2) + getRmax(1) * getTanTheta(2));
  double zstep3 = (getZoff(1) + getRmax(0) * getTanTheta(1));
  double zstep2 = (getZoff(0) + getRmax(0) * getTanTheta(0));
  double zstep1 = (getZoff(0) + getRin() * getTanTheta(0));
  double rout = getRout();
  double rout1 = getRmax(3);
  double rin = getRin();
  double rmid1 = getRmax(0);
  double rmid2 = getRmax(1);
  double rmid3 = (getZoff(4) - getZoff(2)) / getTanTheta(2);
  double rmid4 = getRmax(2);

  std::vector<double> pgonZ = {-zmax,
                               -zstep5,
                               -zstep5,
                               -zstep4,
                               -zstep3,
                               -zstep2,
                               -zstep1,
                               0,
                               zstep1,
                               zstep2,
                               zstep3,
                               zstep4,
                               zstep5,
                               zstep5,
                               zmax};

  std::vector<double> pgonRmin = {
      rmid4, rmid3, rmid3, rmid2, rmid1, rmid1, rin, rin, rin, rmid1, rmid1, rmid2, rmid3, rmid3, rmid4};

  std::vector<double> pgonRmax = {
      rout1, rout1, rout, rout, rout, rout, rout, rout, rout, rout, rout, rout, rout, rout1, rout1};

  std::vector<double> pgonZHalf = {0, zstep1, zstep2, zstep3, zstep4, zstep5, zstep5, zmax};

  std::vector<double> pgonRminHalf = {rin, rin, rmid1, rmid1, rmid2, rmid3, rmid3, rmid4};

  std::vector<double> pgonRmaxHalf = {rout, rout, rout, rout, rout, rout, rout1, rout1};

  std::string name("Null");
  DDSolid solid;
  if (nf == 0) {
    solid = DDSolidFactory::polyhedra(
        DDName(idName, idNameSpace), getNsectortot(), -alpha, dphi, pgonZ, pgonRmin, pgonRmax);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << DDName(idName, idNameSpace) << " Polyhedra made of "
                                 << getGenMaterial() << " with " << getNsectortot() << " sectors from "
                                 << convertRadToDeg(-alpha) << " to " << convertRadToDeg(-alpha + dphi) << " and with "
                                 << nsec << " sections ";
    for (unsigned int i = 0; i < pgonZ.size(); i++)
      edm::LogVerbatim("HCalGeom") << "\t"
                                   << "\tZ = " << pgonZ[i] << "\tRmin = " << pgonRmin[i] << "\tRmax = " << pgonRmax[i];
#endif
  } else {
    solid = DDSolidFactory::polyhedra(
        DDName(idName, idNameSpace), getNsectortot(), -alpha, dphi, pgonZHalf, pgonRminHalf, pgonRmaxHalf);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << DDName(idName, idNameSpace) << " Polyhedra made of "
                                 << getGenMaterial() << " with " << getNsectortot() << " sectors from "
                                 << convertRadToDeg(-alpha) << " to " << convertRadToDeg(-alpha + dphi) << " and with "
                                 << nsec << " sections ";
    for (unsigned int i = 0; i < pgonZHalf.size(); i++)
      edm::LogVerbatim("HCalGeom") << "\t"
                                   << "\tZ = " << pgonZHalf[i] << "\tRmin = " << pgonRminHalf[i]
                                   << "\tRmax = " << pgonRmaxHalf[i];
#endif
  }

  DDName matname(DDSplit(getGenMaterial()).first, DDSplit(getGenMaterial()).second);
  DDMaterial matter(matname);
  DDLogicalPart genlogic(DDName(idName, idNameSpace), matter, solid);

  DDName parentName = parent().name();
  DDTranslation r0(0, 0, 0);
  cpv.position(DDName(idName, idNameSpace), parentName, 1, r0, rot);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << DDName(idName, idNameSpace) << " number 1 positioned in "
                               << parentName << " at (0, 0, 0) with no rotation";
#endif
  //Forward and backwards halfs
  name = idName + "Half";
  nf = (ntot + 1) / 2;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << DDName(name, idNameSpace) << " Polyhedra made of "
                               << getGenMaterial() << " with " << getNsectortot() << " sectors from "
                               << convertRadToDeg(-alpha) << " to " << convertRadToDeg(-alpha + dphi) << " and with "
                               << nf << " sections ";
  for (unsigned int i = 0; i < pgonZHalf.size(); i++)
    edm::LogVerbatim("HCalGeom") << "\t"
                                 << "\tZ = " << pgonZHalf[i] << "\tRmin = " << pgonRminHalf[i]
                                 << "\tRmax = " << pgonRmaxHalf[i];
#endif

  solid = DDSolidFactory::polyhedra(
      DDName(name, idNameSpace), getNsectortot(), -alpha, dphi, pgonZHalf, pgonRminHalf, pgonRmaxHalf);
  DDLogicalPart genlogich(DDName(name, idNameSpace), matter, solid);

  cpv.position(genlogich, genlogic, 1, r0, rot);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << genlogich.name() << " number 1 positioned in "
                               << genlogic.name() << " at (0, 0, 0) with no rotation";
#endif
  if (getNhalf() != 1) {
    rot = DDRotation(DDName(rotHalf, rotns));
    cpv.position(genlogich, genlogic, 2, r0, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << genlogich.name() << " number 2 positioned in "
                                 << genlogic.name() << " at " << r0 << " with " << rot;
#endif
  }  //end if (getNhalf...

  //Construct sector (from -alpha to +alpha)
  name = idName + "Module";
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << DDName(name, idNameSpace) << " Polyhedra made of "
                               << getGenMaterial() << " with 1 sector from " << convertRadToDeg(-alpha) << " to "
                               << convertRadToDeg(alpha) << " and with " << nf << " sections";
  for (unsigned int i = 0; i < pgonZHalf.size(); i++)
    edm::LogVerbatim("HCalGeom") << "\t"
                                 << "\tZ = " << pgonZHalf[i] << "\tRmin = " << pgonRminHalf[i]
                                 << "\tRmax = " << pgonRmaxHalf[i];
#endif

  solid =
      DDSolidFactory::polyhedra(DDName(name, idNameSpace), 1, -alpha, 2 * alpha, pgonZHalf, pgonRminHalf, pgonRmaxHalf);
  DDLogicalPart seclogic(DDName(name, idNameSpace), matter, solid);

  double theta = 90._deg;
  for (int ii = 0; ii < getNsectortot(); ii++) {
    double phi = ii * 2 * alpha;
    double phiy = phi + 90._deg;

    DDRotation rotation;
    std::string rotstr("NULL");
    if (phi != 0) {
      rotstr = "R" + formatAsDegreesInInteger(phi);
      rotation = DDRotation(DDName(rotstr, rotns));
      if (!rotation) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Creating a new "
                                     << "rotation " << rotstr << "\t 90," << convertRadToDeg(phi) << ",90,"
                                     << (90 + convertRadToDeg(phi)) << ", 0, 0";
#endif
        rotation = DDrot(DDName(rotstr, rotns), theta, phi, theta, phiy, 0, 0);
      }  //if !rotation
    }    //if phideg!=0

    cpv.position(seclogic, genlogich, ii + 1, r0, rotation);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << seclogic.name() << " number " << ii + 1 << " positioned in "
                                 << genlogich.name() << " at " << r0 << " with " << rotation;
#endif
  }

  //Construct the things inside the sector
  constructInsideSector(seclogic, cpv);
}

void DDHCalBarrelAlgo::constructInsideSector(const DDLogicalPart& sector, DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Layers (" << getNLayers() << ") ...";
#endif
  double alpha = (1._pi) / getNsectors();
  double rin = getRin();
  for (int i = 0; i < getNLayers(); i++) {
    std::string name = idName + getLayerLabel(i);
    DDName matname(DDSplit(getLayerMaterial(i)).first,
                   DDSplit(getLayerMaterial(i)).second);  //idNameSpace);
    DDMaterial matter(matname);

    double width = getLayerWidth(i);
    double rout = rin + width;

    int in = 0, out = 0;
    for (int j = 0; j < getRzones() - 1; j++) {
      if (rin >= getRmax(j))
        in = j + 1;
      if (rout > getRmax(j))
        out = j + 1;
    }
    double zout = getZoff(in) + rin * getTanTheta(in);

    //!!!!!!!!!!!!!!!!!Should be zero. And removed as soon as
    //vertical walls are allowed in SolidPolyhedra
    double deltaz = 0;
#ifdef EDM_ML_DEBUG
    int nsec = 2;
#endif
    std::vector<double> pgonZ, pgonRmin, pgonRmax;
    // index 0
    pgonZ.emplace_back(0);
    pgonRmin.emplace_back(rin);
    pgonRmax.emplace_back(rout);
    // index 1
    pgonZ.emplace_back(zout);
    pgonRmin.emplace_back(rin);
    pgonRmax.emplace_back(rout);
    if (in == out) {
      if (in <= 3) {
        //index 2
        pgonZ.emplace_back(getZoff(in) + rout * getTanTheta(in));
        pgonRmin.emplace_back(pgonRmax[1]);
        pgonRmax.emplace_back(pgonRmax[1]);
#ifdef EDM_ML_DEBUG
        nsec++;
#endif
      }
    } else {
      if (in == 3) {
        //redo index 1, add index 2
        pgonZ[1] = (getZoff(out) + getRmax(out) * getTanTheta(out));
        pgonZ.emplace_back(pgonZ[1] + deltaz);
        pgonRmin.emplace_back(pgonRmin[1]);
        pgonRmax.emplace_back(getRmax(in));
        //index 3
        pgonZ.emplace_back(getZoff(in) + getRmax(in) * getTanTheta(in));
        pgonRmin.emplace_back(pgonRmin[2]);
        pgonRmax.emplace_back(pgonRmax[2]);
#ifdef EDM_ML_DEBUG
        nsec += 2;
#endif
      } else {
        //index 2
        pgonZ.emplace_back(getZoff(in) + getRmax(in) * getTanTheta(in));
        pgonRmin.emplace_back(getRmax(in));
        pgonRmax.emplace_back(pgonRmax[1]);
#ifdef EDM_ML_DEBUG
        nsec++;
#endif
        if (in == 0) {
          pgonZ.emplace_back(getZoff(out) + getRmax(in) * getTanTheta(out));
          pgonRmin.emplace_back(pgonRmin[2]);
          pgonRmax.emplace_back(pgonRmax[2]);
#ifdef EDM_ML_DEBUG
          nsec++;
#endif
        }
        if (in <= 1) {
          pgonZ.emplace_back(getZoff(out) + rout * getTanTheta(out));
          pgonRmin.emplace_back(rout);
          pgonRmax.emplace_back(rout);
#ifdef EDM_ML_DEBUG
          nsec++;
#endif
        }
      }
    }
    //Solid & volume
    DDSolid solid;
    double alpha1 = alpha;
    if (getLayerGap(i) > 1.e-6) {
      double rmid = 0.5 * (rin + rout);
      double width = rmid * tan(alpha) - getLayerGap(i);
      alpha1 = atan(width / rmid);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "\t"
                                   << "Alpha_1 modified from " << convertRadToDeg(alpha) << " to "
                                   << convertRadToDeg(alpha1) << " Rmid " << rmid << " Reduced width " << width;
#endif
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << name << " (Layer " << i << ") Polyhedra made of "
                                 << getLayerMaterial(i) << " with 1 sector from " << convertRadToDeg(-alpha1) << " to "
                                 << convertRadToDeg(alpha1) << " and with " << nsec << " sections";
    for (unsigned int k = 0; k < pgonZ.size(); k++)
      edm::LogVerbatim("HCalGeom") << "\t"
                                   << "\t" << pgonZ[k] << "\t" << pgonRmin[k] << "\t" << pgonRmax[k];
#endif
    solid = DDSolidFactory::polyhedra(DDName(name, idNameSpace), 1, -alpha1, 2 * alpha1, pgonZ, pgonRmin, pgonRmax);
    DDLogicalPart glog(DDName(name, idNameSpace), matter, solid);

    cpv.position(glog, sector, getLayerId(i), DDTranslation(0.0, 0.0, 0.0), DDRotation());
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " number " << getLayerId(i)
                                 << " positioned in " << sector.name() << " at (0,0,0) with no rotation";
#endif
    constructInsideLayers(glog,
                          getLayerLabel(i),
                          getLayerId(i),
                          getLayerAbsorb(i),
                          rin,
                          getLayerD1(i),
                          alpha1,
                          getLayerD2(i),
                          getLayerAlpha(i),
                          getLayerT1(i),
                          getLayerT2(i),
                          cpv);
    rin = rout;
  }
}

void DDHCalBarrelAlgo::constructInsideLayers(const DDLogicalPart& laylog,
                                             const std::string& nm,
                                             int id,
                                             int nAbs,
                                             double rin,
                                             double d1,
                                             double alpha1,
                                             double d2,
                                             double alpha2,
                                             double t1,
                                             double t2,
                                             DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: \t\tInside layer " << id << "...";
#endif
  ///////////////////////////////////////////////////////////////
  //Pointers to the Rotation Matrices and to the Materials
  DDRotation rot(DDName(detRot, rotns));

  std::string nam0 = nm + "In";
  std::string name = idName + nam0;
  DDName matName(DDSplit(getDetMat()).first, DDSplit(getDetMat()).second);
  DDMaterial matter(matName);

  DDSolid solid;
  DDLogicalPart glog, mother;
  double rsi, dx, dy, dz, x, y;
  int i, in;
  //Two lower volumes
  if (alpha1 > 0) {
    rsi = rin + d1;
    in = 0;
    for (i = 0; i < getRzones() - 1; i++) {
      if (rsi >= getRmax(i))
        in = i + 1;
    }
    dx = 0.5 * t1;
    dy = 0.5 * rsi * (tan(alpha1) - tan(alpha2));
    dz = 0.5 * (getZoff(in) + rsi * getTanTheta(in));
    x = rsi + dx;
    y = 0.5 * rsi * (tan(alpha1) + tan(alpha2));
    DDTranslation r11(x, y, dz);
    DDTranslation r12(x, -y, dz);

    solid = DDSolidFactory::box(DDName(name + "1", idNameSpace), dx, dy, dz);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Box made of " << getDetMat()
                                 << " of dimensions " << dx << ", " << dy << ", " << dz;
#endif
    glog = DDLogicalPart(solid.ddname(), matter, solid);

    if (nAbs != 0) {
      mother = constructSideLayer(laylog, name, nAbs, rin, alpha1, cpv);
    } else {
      mother = laylog;
    }
    cpv.position(glog, mother, idOffset + 1, r11, DDRotation());
    cpv.position(glog, mother, idOffset + 2, r12, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " Number " << idOffset + 1
                                 << " positioned in " << mother.name() << " at " << r11 << " with no rotation\n"
                                 << "DDHCalBarrelAlgo: " << glog.name() << " Number " << idOffset + 2
                                 << " positioned in " << mother.name() << " at " << r12 << " with " << rot;
#endif
    //Constructin the plastics and scintillators inside
    constructInsideDetectors(glog, nam0 + "1", id, dx, dy, dz, 1, cpv);
  }

  //Upper volume
  rsi = rin + d2;
  in = 0;
  for (i = 0; i < getRzones() - 1; i++) {
    if (rsi >= getRmax(i))
      in = i + 1;
  }
  dx = 0.5 * t2;
  dy = 0.5 * rsi * tan(alpha2);
  dz = 0.5 * (getZoff(in) + rsi * getTanTheta(in));
  x = rsi + dx;
  DDTranslation r21(x, dy, dz);
  DDTranslation r22(x, -dy, dz);

  solid = DDSolidFactory::box(DDName(name + "2", idNameSpace), dx, dy, dz);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Box made of " << getDetMat()
                               << " of dimensions " << dx << ", " << dy << ", " << dz;
#endif
  glog = DDLogicalPart(solid.ddname(), matter, solid);

  if (nAbs < 0) {
    mother = constructMidLayer(laylog, name, rin, alpha1, cpv);
  } else {
    mother = laylog;
  }
  cpv.position(glog, mother, idOffset + 3, r21, DDRotation());
  cpv.position(glog, mother, idOffset + 4, r22, rot);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " Number " << idOffset + 3 << " positioned in "
                               << mother.name() << " at " << r21
                               << " with no rotation\nDDHCalBarrelAlgo: " << glog.name() << " Number " << idOffset + 4
                               << " positioned in " << mother.name() << " at " << r22 << " with " << rot;
#endif
  //Constructin the plastics and scintillators inside
  constructInsideDetectors(glog, nam0 + "2", id, dx, dy, dz, 2, cpv);
}

DDLogicalPart DDHCalBarrelAlgo::constructSideLayer(
    const DDLogicalPart& laylog, const std::string& nm, int nAbs, double rin, double alpha, DDCompactView& cpv) {
  //Extra absorber layer
  int k = abs(nAbs) - 1;
  std::string namek = nm + "Side";
  double rsi = rin + getSideD(k);
  int in = 0;
  for (int i = 0; i < getRzones() - 1; i++) {
    if (rsi >= getRmax(i))
      in = i + 1;
  }
  std::vector<double> pgonZ, pgonRmin, pgonRmax;
  // index 0
  pgonZ.emplace_back(0.0);
  pgonRmin.emplace_back(rsi);
  pgonRmax.emplace_back(rsi + getSideT(k));
  // index 1
  pgonZ.emplace_back(getZoff(in) + rsi * getTanTheta(in));
  pgonRmin.emplace_back(rsi);
  pgonRmax.emplace_back(pgonRmax[0]);
  // index 2
  pgonZ.emplace_back(getZoff(in) + pgonRmax[0] * getTanTheta(in));
  pgonRmin.emplace_back(pgonRmax[1]);
  pgonRmax.emplace_back(pgonRmax[1]);
  DDSolid solid =
      DDSolidFactory::polyhedra(DDName(namek, idNameSpace), 1, -alpha, 2 * alpha, pgonZ, pgonRmin, pgonRmax);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of " << getSideMat(k)
                               << " with 1 sector from " << convertRadToDeg(-alpha) << " to " << convertRadToDeg(alpha)
                               << " and with " << pgonZ.size() << " sections";
  for (unsigned int ii = 0; ii < pgonZ.size(); ii++)
    edm::LogVerbatim("HCalGeom") << "\t\tZ = " << pgonZ[ii] << "\tRmin = " << pgonRmin[ii]
                                 << "\tRmax = " << pgonRmax[ii];
#endif

  DDName matName(DDSplit(getSideMat(k)).first, DDSplit(getSideMat(k)).second);
  DDMaterial matter(matName);
  DDLogicalPart glog = DDLogicalPart(solid.ddname(), matter, solid);

  cpv.position(glog, laylog, 1, DDTranslation(), DDRotation());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " Number 1 positioned in " << laylog.name()
                               << " at (0,0,0) with no rotation";
#endif
  if (nAbs < 0) {
    DDLogicalPart mother = glog;
    double rmid = pgonRmax[0];
    for (int i = 0; i < getSideAbsorber(); i++) {
      double alpha1 = atan(getSideAbsW(i) / rmid);
      if (alpha1 > 0) {
        std::string name = namek + getSideAbsName(i);
        solid = DDSolidFactory::polyhedra(DDName(name, idNameSpace), 1, -alpha1, 2 * alpha1, pgonZ, pgonRmin, pgonRmax);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of "
                                     << getSideAbsMat(i) << " with 1 sector from " << convertRadToDeg(-alpha1) << " to "
                                     << convertRadToDeg(alpha1) << " and with " << pgonZ.size() << " sections";
        for (unsigned int ii = 0; ii < pgonZ.size(); ii++)
          edm::LogVerbatim("HCalGeom") << "\t\tZ = " << pgonZ[ii] << "\tRmin = " << pgonRmin[ii]
                                       << "\tRmax = " << pgonRmax[ii];
#endif

        DDName matName(DDSplit(getSideAbsMat(i)).first, DDSplit(getSideAbsMat(i)).second);
        DDMaterial matter(matName);
        DDLogicalPart log = DDLogicalPart(solid.ddname(), matter, solid);

        cpv.position(log, mother, 1, DDTranslation(), DDRotation());
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << log.name() << " Number 1 positioned in "
                                     << mother.name() << " at (0,0,0) with no rotation";
#endif
        mother = log;
      }
    }
  }
  return glog;
}

DDLogicalPart DDHCalBarrelAlgo::constructMidLayer(
    const DDLogicalPart& laylog, const std::string& nm, double rin, double alpha, DDCompactView& cpv) {
  DDSolid solid;
  DDLogicalPart log, glog;
  std::string name = nm + "Mid";
  for (int k = 0; k < getAbsorberN(); k++) {
    std::string namek = name + getAbsorbName(k);
    double rsi = rin + getAbsorbD(k);
    int in = 0;
    for (int i = 0; i < getRzones() - 1; i++) {
      if (rsi >= getRmax(i))
        in = i + 1;
    }
    std::vector<double> pgonZ, pgonRmin, pgonRmax;
    // index 0
    pgonZ.emplace_back(0.0);
    pgonRmin.emplace_back(rsi);
    pgonRmax.emplace_back(rsi + getAbsorbT(k));
    // index 1
    pgonZ.emplace_back(getZoff(in) + rsi * getTanTheta(in));
    pgonRmin.emplace_back(rsi);
    pgonRmax.emplace_back(pgonRmax[0]);
    // index 2
    pgonZ.emplace_back(getZoff(in) + pgonRmax[0] * getTanTheta(in));
    pgonRmin.emplace_back(pgonRmax[1]);
    pgonRmax.emplace_back(pgonRmax[1]);
    solid = DDSolidFactory::polyhedra(DDName(namek, idNameSpace), 1, -alpha, 2 * alpha, pgonZ, pgonRmin, pgonRmax);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of " << getAbsorbMat(k)
                                 << " with 1 sector from " << convertRadToDeg(-alpha) << " to "
                                 << convertRadToDeg(alpha) << " and with " << pgonZ.size() << " sections";
    for (unsigned int ii = 0; ii < pgonZ.size(); ii++)
      edm::LogVerbatim("HCalGeom") << "\t\tZ = " << pgonZ[ii] << "\tRmin = " << pgonRmin[ii]
                                   << "\tRmax = " << pgonRmax[ii];
#endif

    DDName matName(DDSplit(getAbsorbMat(k)).first, DDSplit(getAbsorbMat(k)).second);
    DDMaterial matter(matName);
    log = DDLogicalPart(solid.ddname(), matter, solid);

    cpv.position(log, laylog, 1, DDTranslation(), DDRotation());
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << log.name() << " Number 1 positioned in " << laylog.name()
                                 << " at (0,0,0) with no rotation";
#endif
    if (k == 0) {
      double rmin = pgonRmin[0];
      double rmax = pgonRmax[0];
      DDLogicalPart mother = log;
      for (int i = 0; i < 1; i++) {
        double alpha1 = atan(getMidAbsW(i) / rmin);
        std::string namek = name + getMidAbsName(i);
        solid =
            DDSolidFactory::polyhedra(DDName(namek, idNameSpace), 1, -alpha1, 2 * alpha1, pgonZ, pgonRmin, pgonRmax);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of " << getMidAbsMat(i)
                                     << " with 1 sector from " << convertRadToDeg(-alpha1) << " to "
                                     << convertRadToDeg(alpha1) << " and with " << pgonZ.size() << " sections";
        for (unsigned int ii = 0; ii < pgonZ.size(); ii++)
          edm::LogVerbatim("HCalGeom") << "\t\tZ = " << pgonZ[ii] << "\tRmin = " << pgonRmin[ii]
                                       << "\tRmax = " << pgonRmax[ii];
#endif

        DDName matNam1(DDSplit(getMidAbsMat(i)).first, DDSplit(getMidAbsMat(i)).second);
        DDMaterial matter1(matNam1);
        log = DDLogicalPart(solid.ddname(), matter1, solid);

        cpv.position(log, mother, 1, DDTranslation(), DDRotation());
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << log.name() << " Number 1 positioned in "
                                     << mother.name() << " at (0,0,0) with no rotation";
#endif
        mother = log;
      }

      // Now the layer with detectors
      double rmid = rmin + getMiddleD();
      pgonRmin[0] = rmid;
      pgonRmax[0] = rmax;
      pgonRmin[1] = rmid;
      pgonRmax[1] = rmax;
      pgonZ[1] = getZoff(in) + rmid * getTanTheta(in);
      pgonRmin[2] = rmax;
      pgonRmax[2] = rmax;
      pgonZ[2] = getZoff(in) + rmax * getTanTheta(in);
      double alpha1 = atan(getMiddleW() / rmin);
      solid = DDSolidFactory::polyhedra(DDName(name, idNameSpace), 1, -alpha1, 2 * alpha1, pgonZ, pgonRmin, pgonRmax);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of " << getMiddleMat()
                                   << " with 1 sector from " << convertRadToDeg(-alpha1) << " to "
                                   << convertRadToDeg(alpha1) << " and with " << pgonZ.size() << " sections";
      for (unsigned int ii = 0; ii < pgonZ.size(); ii++)
        edm::LogVerbatim("HCalGeom") << "\t\tZ = " << pgonZ[ii] << "\tRmin = " << pgonRmin[ii]
                                     << "\tRmax = " << pgonRmax[ii];
#endif

      DDName matNam1(DDSplit(getMiddleMat()).first, DDSplit(getMiddleMat()).second);
      DDMaterial matter1(matNam1);
      glog = DDLogicalPart(solid.ddname(), matter1, solid);

      cpv.position(glog, mother, 1, DDTranslation(), DDRotation());
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " Number 1 positioned in " << mother.name()
                                   << " at (0,0,0) with no rotation";
#endif
      // Now the remaining absorber layers
      for (int i = 1; i < getMidAbsorber(); i++) {
        namek = name + getMidAbsName(i);
        rmid = rmin + getMidAbsT(i);
        pgonRmin[0] = rmin;
        pgonRmax[0] = rmid;
        pgonRmin[1] = rmin;
        pgonRmax[1] = rmid;
        pgonZ[1] = getZoff(in) + rmin * getTanTheta(in);
        pgonRmin[2] = rmid;
        pgonRmax[2] = rmid;
        pgonZ[2] = getZoff(in) + rmid * getTanTheta(in);
        alpha1 = atan(getMidAbsW(i) / rmin);
        solid =
            DDSolidFactory::polyhedra(DDName(namek, idNameSpace), 1, -alpha1, 2 * alpha1, pgonZ, pgonRmin, pgonRmax);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of " << getMidAbsMat(i)
                                     << " with 1 sector from " << convertRadToDeg(-alpha1) << " to "
                                     << convertRadToDeg(alpha1) << " and with " << pgonZ.size() << " sections";
        for (unsigned int ii = 0; ii < pgonZ.size(); ii++)
          edm::LogVerbatim("HCalGeom") << "\t\tZ = " << pgonZ[ii] << "\tRmin = " << pgonRmin[ii]
                                       << "\tRmax = " << pgonRmax[ii];
#endif

        DDName matName2(DDSplit(getMidAbsMat(i)).first, DDSplit(getMidAbsMat(i)).second);
        DDMaterial matter2(matName2);
        log = DDLogicalPart(solid.ddname(), matter2, solid);

        cpv.position(log, mother, i, DDTranslation(), DDRotation());
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << log.name() << " Number " << i << " positioned in "
                                     << mother.name() << " at (0,0,0) with no "
                                     << "rotation";
#endif
        mother = log;
      }
    }
  }
  return glog;
}

void DDHCalBarrelAlgo::constructInsideDetectors(const DDLogicalPart& detector,
                                                const std::string& name,
                                                int id,
                                                double dx,
                                                double dy,
                                                double dz,
                                                int type,
                                                DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: \t\tInside detector " << id << "...";
#endif

  DDName plmatname(DDSplit(getDetMatPl()).first, DDSplit(getDetMatPl()).second);
  DDMaterial plmatter(plmatname);
  DDName scmatname(DDSplit(getDetMatSc()).first, DDSplit(getDetMatSc()).second);
  DDMaterial scmatter(scmatname);

  std::string plname = detector.name().name() + "Plastic_";
  std::string scname = idName + "Scintillator" + name;

  id--;
  DDSolid solid;
  DDLogicalPart glog;
  double wid, y = 0;
  double dx1, dx2, shiftX;

  if (type == 1) {
    wid = 0.5 * getDetWidth1(id);
    if (getDetPosY(id) > 0)
      y = -dy + wid;
    dx1 = 0.5 * getDetT11(id);
    dx2 = 0.5 * getDetT21(id);
    shiftX = getDetdP1(id);
  } else {
    wid = 0.5 * getDetWidth2(id);
    dx1 = 0.5 * getDetT12(id);
    dx2 = 0.5 * getDetT22(id);
    shiftX = getDetdP2(id);
  }

  solid = DDSolidFactory::box(DDName(plname + "1", idNameSpace), dx1, wid, dz);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Box made of " << getDetMatPl()
                               << " of dimensions " << dx1 << ", " << wid << ", " << dz;
#endif
  glog = DDLogicalPart(solid.ddname(), plmatter, solid);

  double x = shiftX + dx1 - dx;
  cpv.position(glog, detector, 1, DDTranslation(x, y, 0), DDRotation());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " Number 1 positioned in " << detector.name()
                               << " at (" << x << "," << y << ",0) with no rotation";
#endif
  solid = DDSolidFactory::box(DDName(scname, idNameSpace), 0.5 * getDetTsc(id), wid, dz);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Box made of " << getDetMatSc()
                               << " of dimensions " << 0.5 * getDetTsc(id) << ", " << wid << ", " << dz;
#endif
  glog = DDLogicalPart(solid.ddname(), scmatter, solid);

  x += dx1 + 0.5 * getDetTsc(id);
  int copyNo = id * 10 + getDetType(id);
  cpv.position(glog, detector, copyNo, DDTranslation(x, y, 0), DDRotation());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " Number " << copyNo << " positioned in "
                               << detector.name() << " at (" << x << "," << y << ",0) with no rotation";
#endif
  solid = DDSolidFactory::box(DDName(plname + "2", idNameSpace), dx2, wid, dz);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Box made of " << getDetMatPl()
                               << " of dimensions " << dx2 << ", " << wid << ", " << dz;
#endif
  glog = DDLogicalPart(solid.ddname(), plmatter, solid);

  x += 0.5 * getDetTsc(id) + dx2;
  cpv.position(glog, detector, 1, DDTranslation(x, y, 0), DDRotation());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " Number 1 positioned in " << detector.name()
                               << " at (" << x << "," << y << ",0) with no rotation";
#endif
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHCalBarrelAlgo, "hcal:DDHCalBarrelAlgo");
