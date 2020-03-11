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

#include "DataFormats/Math/interface/GeantUnits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DD4hep/DetFactoryHelper.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

struct HcalBarrelAlgo {
  //General Volume
  //      <----- Zmax ------>
  //Router************************-------
  //      *                      *Rstep2|        Theta angle w.r.t. vertical
  //      *                      *---------------
  //      *                     *               |
  //      *                    *Theta[i]        Rmax[i]
  //      *                   *---------------  |
  //                        *Theta[0] Rmax[0]|  |
  //Rinner*****************----------------------

  std::string genMaterial;     //General material
  int nsectors;                //Number of potenital straight edges
  int nsectortot;              //Number of straight edges (actual)
  int nhalf;                   //Number of half modules
  double rinner, router;       //See picture
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

  HcalBarrelAlgo() = delete;

  HcalBarrelAlgo(cms::DDParsingContext& ctxt, xml_h& e) {
    cms::DDNamespace ns(ctxt, e, true);
    cms::DDAlgoArguments args(ctxt, e);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Creating an instance";
#endif

    genMaterial = args.value<std::string>("MaterialName");
    nsectors = args.value<int>("NSector");
    nsectortot = args.value<int>("NSectorTot");
    nhalf = args.value<int>("NHalf");
    rinner = args.value<double>("RIn");
    router = args.value<double>("ROut");
    rzones = args.value<int>("RZones");
    rotHalf = args.value<std::string>("RotHalf");
    rotns = args.value<std::string>("RotNameSpace");

    theta = args.value<std::vector<double> >("Theta");
    rmax = args.value<std::vector<double> >("RMax");
    zoff = args.value<std::vector<double> >("ZOff");
    for (int i = 0; i < rzones; i++) {
      ttheta.emplace_back(tan(theta[i]));  //*deg already done in XML
    }
    if (rzones > 3)
      rmax[2] = (zoff[3] - zoff[2]) / ttheta[2];

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: General material " << genMaterial << "\tSectors " << nsectors
                                 << ", " << nsectortot << "\tHalves " << nhalf << "\tRotation matrix " << rotns << ":"
                                 << rotHalf << "\n\t\t" << convertCmToMm(rinner) << "\t" << convertCmToMm(router)
                                 << "\t" << rzones;
    for (int i = 0; i < rzones; i++)
      edm::LogVerbatim("HCalGeom") << "\tTheta[" << i << "] = " << theta[i] << "\trmax[" << i
                                   << "] = " << convertCmToMm(rmax[i]) << "\tzoff[" << i
                                   << "] = " << convertCmToMm(zoff[i]);
#endif
    ///////////////////////////////////////////////////////////////
    //Layers
    nLayers = args.value<int>("NLayers");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Layer\t" << nLayers;
#endif
    layerId = args.value<std::vector<int> >("Id");
    layerLabel = args.value<std::vector<std::string> >("LayerLabel");
    layerMat = args.value<std::vector<std::string> >("LayerMat");
    layerWidth = args.value<std::vector<double> >("LayerWidth");
    layerD1 = args.value<std::vector<double> >("D1");
    layerD2 = args.value<std::vector<double> >("D2");
    layerAlpha = args.value<std::vector<double> >("Alpha2");
    layerT1 = args.value<std::vector<double> >("T1");
    layerT2 = args.value<std::vector<double> >("T2");
    layerAbsorb = args.value<std::vector<int> >("AbsL");
    layerGap = args.value<std::vector<double> >("Gap");
#ifdef EDM_ML_DEBUG
    for (int i = 0; i < nLayers; i++)
      edm::LogVerbatim("HCalGeom") << layerLabel[i] << "\t" << layerId[i] << "\t" << layerMat[i] << "\t"
                                   << convertCmToMm(layerWidth[i]) << "\t" << convertCmToMm(layerD1[i]) << "\t"
                                   << convertCmToMm(layerD2[i]) << "\t" << layerAlpha[i] << "\t"
                                   << convertCmToMm(layerT1[i]) << "\t" << convertCmToMm(layerT2[i]) << "\t"
                                   << layerAbsorb[i] << "\t" << convertCmToMm(layerGap[i]);
#endif

    ///////////////////////////////////////////////////////////////
    //Absorber Layers and middle part
    absorbName = args.value<std::vector<std::string> >("AbsorbName");
    absorbMat = args.value<std::vector<std::string> >("AbsorbMat");
    absorbD = args.value<std::vector<double> >("AbsorbD");
    absorbT = args.value<std::vector<double> >("AbsorbT");
    nAbsorber = absorbName.size();
#ifdef EDM_ML_DEBUG
    for (int i = 0; i < nAbsorber; i++)
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << absorbName[i] << " Material " << absorbMat[i] << " d "
                                   << convertCmToMm(absorbD[i]) << " t " << convertCmToMm(absorbT[i]);
#endif
    middleMat = args.value<std::string>("MiddleMat");
    middleD = args.value<double>("MiddleD");
    middleW = args.value<double>("MiddleW");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Middle material " << middleMat << " d " << convertCmToMm(middleD)
                                 << " w " << convertCmToMm(middleW);
#endif
    midName = args.value<std::vector<std::string> >("MidAbsName");
    midMat = args.value<std::vector<std::string> >("MidAbsMat");
    midW = args.value<std::vector<double> >("MidAbsW");
    midT = args.value<std::vector<double> >("MidAbsT");
    nMidAbs = midName.size();
#ifdef EDM_ML_DEBUG
    for (int i = 0; i < nMidAbs; i++)
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << midName[i] << " Material " << midMat[i] << " W "
                                   << convertCmToMm(midW[i]) << " T " << convertCmToMm(midT[i]);
#endif

    //Absorber layers in the side part
    sideMat = args.value<std::vector<std::string> >("SideMat");
    sideD = args.value<std::vector<double> >("SideD");
    sideT = args.value<std::vector<double> >("SideT");
#ifdef EDM_ML_DEBUG
    for (unsigned int i = 0; i < sideMat.size(); i++)
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Side material " << sideMat[i] << " d "
                                   << convertCmToMm(sideD[i]) << " t " << convertCmToMm(sideT[i]);
#endif
    sideAbsName = args.value<std::vector<std::string> >("SideAbsName");
    sideAbsMat = args.value<std::vector<std::string> >("SideAbsMat");
    sideAbsW = args.value<std::vector<double> >("SideAbsW");
    nSideAbs = sideAbsName.size();
#ifdef EDM_ML_DEBUG
    for (int i = 0; i < nSideAbs; i++)
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << sideAbsName[i] << " Material " << sideAbsMat[i] << " W "
                                   << convertCmToMm(sideAbsW[i]);
#endif

    ///////////////////////////////////////////////////////////////
    // Detectors

    detMat = args.value<std::string>("DetMat");
    detRot = args.value<std::string>("DetRot");
    detMatPl = args.value<std::string>("DetMatPl");
    detMatSc = args.value<std::string>("DetMatSc");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Detector (" << nLayers << ") Rotation matrix " << rotns << ":"
                                 << detRot << "\n\t\t" << detMat << "\t" << detMatPl << "\t" << detMatSc;
#endif
    detType = args.value<std::vector<int> >("DetType");
    detdP1 = args.value<std::vector<double> >("DetdP1");
    detdP2 = args.value<std::vector<double> >("DetdP2");
    detT11 = args.value<std::vector<double> >("DetT11");
    detT12 = args.value<std::vector<double> >("DetT12");
    detTsc = args.value<std::vector<double> >("DetTsc");
    detT21 = args.value<std::vector<double> >("DetT21");
    detT22 = args.value<std::vector<double> >("DetT22");
    detWidth1 = args.value<std::vector<double> >("DetWidth1");
    detWidth2 = args.value<std::vector<double> >("DetWidth2");
    detPosY = args.value<std::vector<int> >("DetPosY");
#ifdef EDM_ML_DEBUG
    for (int i = 0; i < nLayers; i++)
      edm::LogVerbatim("HCalGeom") << i + 1 << "\t" << detType[i] << "\t" << convertCmToMm(detdP1[i]) << ", "
                                   << convertCmToMm(detdP2[i]) << "\t" << convertCmToMm(detT11[i]) << ", "
                                   << convertCmToMm(detT12[i]) << "\t" << convertCmToMm(detTsc[i]) << "\t"
                                   << convertCmToMm(detT21[i]) << ", " << convertCmToMm(detT22[i]) << "\t"
                                   << convertCmToMm(detWidth1[i]) << "\t" << convertCmToMm(detWidth2[i]) << "\t"
                                   << detPosY[i];
#endif

    //  idName = parentName.name();
    idName = args.value<std::string>("MotherName");
    idNameSpace = ns.name();
    idOffset = args.value<int>("IdOffset");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Parent " << args.parentName() << " idName " << idName
                                 << " NameSpace " << idNameSpace << " Offset " << idOffset;
    edm::LogVerbatim("HCalGeom") << "==>> Constructing DDHCalBarrelAlgo...\n"
                                 << "DDHCalBarrelAlgo: General volume...";
#endif

    double alpha = (1._pi) / nsectors;
    double dphi = nsectortot * (2._pi) / nsectors;
    int nsec, ntot(15);
    if (nhalf == 1)
      nsec = 8;
    else
      nsec = 15;
    int nf = ntot - nsec;

    //Calculate zmin... see HCalBarrel.hh picture. For polyhedra
    //Rmin and Rmax are distances to vertex
    double zmax = zoff[3];
    double zstep5 = zoff[4];
    double zstep4 = (zoff[1] + rmax[1] * ttheta[1]);
    if ((zoff[2] + rmax[1] * ttheta[2]) > zstep4)
      zstep4 = (zoff[2] + rmax[1] * ttheta[2]);
    double zstep3 = (zoff[1] + rmax[0] * ttheta[1]);
    double zstep2 = (zoff[0] + rmax[0] * ttheta[0]);
    double zstep1 = (zoff[0] + rinner * ttheta[0]);
    double rout = router;
    double rout1 = rmax[3];
    double rin = rinner;
    double rmid1 = rmax[0];
    double rmid2 = rmax[1];
    double rmid3 = (zoff[4] - zoff[2]) / ttheta[2];
    double rmid4 = rmax[2];

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
    dd4hep::Solid solid;
    if (nf == 0) {
      solid = dd4hep::Polyhedra(ns.prepend(idName), nsectortot, -alpha, dphi, pgonZ, pgonRmin, pgonRmax);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of " << genMaterial
                                   << " with " << nsectortot << " sectors from " << convertRadToDeg(-alpha) << " to "
                                   << convertRadToDeg(-alpha + dphi) << " and with " << nsec << " sections ";
      for (unsigned int i = 0; i < pgonZ.size(); i++)
        edm::LogVerbatim("HCalGeom") << "\t"
                                     << "\tZ = " << convertCmToMm(pgonZ[i]) << "\tRmin = " << convertCmToMm(pgonRmin[i])
                                     << "\tRmax = " << convertCmToMm(pgonRmax[i]);
#endif
    } else {
      solid = dd4hep::Polyhedra(ns.prepend(idName), nsectortot, -alpha, dphi, pgonZHalf, pgonRminHalf, pgonRmaxHalf);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of " << genMaterial
                                   << " with " << nsectortot << " sectors from " << convertRadToDeg(-alpha) << " to "
                                   << convertRadToDeg(-alpha + dphi) << " and with " << nsec << " sections ";
      for (unsigned int i = 0; i < pgonZHalf.size(); i++)
        edm::LogVerbatim("HCalGeom") << "\t"
                                     << "\tZ = " << convertCmToMm(pgonZHalf[i])
                                     << "\tRmin = " << convertCmToMm(pgonRminHalf[i])
                                     << "\tRmax = " << convertCmToMm(pgonRmaxHalf[i]);
#endif
    }

    dd4hep::Material matter = ns.material(genMaterial);
    dd4hep::Volume genlogic(solid.name(), solid, matter);
    dd4hep::Volume parentName = ns.volume(args.parentName());
    parentName.placeVolume(genlogic, 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << genlogic.name() << " number 1 positioned in "
                                 << parentName.name() << " at (0, 0, 0) with no rotation";
#endif
    //Forward and backwards halfs
    name = idName + "Half";
    nf = (ntot + 1) / 2;
    solid = dd4hep::Polyhedra(ns.prepend(name), nsectortot, -alpha, dphi, pgonZHalf, pgonRminHalf, pgonRmaxHalf);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of " << genMaterial
                                 << " with " << nsectortot << " sectors from " << convertRadToDeg(-alpha) << " to "
                                 << convertRadToDeg(-alpha + dphi) << " and with " << nf << " sections ";
    for (unsigned int i = 0; i < pgonZHalf.size(); i++)
      edm::LogVerbatim("HCalGeom") << "\t"
                                   << "\tZ = " << convertCmToMm(pgonZHalf[i])
                                   << "\tRmin = " << convertCmToMm(pgonRminHalf[i])
                                   << "\tRmax = " << convertCmToMm(pgonRmaxHalf[i]);
#endif
    dd4hep::Volume genlogich(solid.name(), solid, matter);
    genlogic.placeVolume(genlogich, 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << genlogich.name() << " number 1 positioned in "
                                 << genlogic.name() << " at (0, 0, 0) with no rotation";
#endif
    if (nhalf != 1) {
      dd4hep::Rotation3D rot = getRotation(rotHalf, rotns, ns);
      genlogic.placeVolume(genlogich, 2, rot);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << genlogich.name() << " number 2 positioned in "
                                   << genlogic.name() << " at (0, 0, 0) with " << rot;
#endif
    }  //end if (getNhalf...

    //Construct sector (from -alpha to +alpha)
    name = idName + "Module";
    solid = dd4hep::Polyhedra(ns.prepend(name), 1, -alpha, 2 * alpha, pgonZHalf, pgonRminHalf, pgonRmaxHalf);
    dd4hep::Volume seclogic(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of " << matter.name()
                                 << " with 1 sector from " << convertRadToDeg(-alpha) << " to "
                                 << convertRadToDeg(alpha) << " and with " << nf << " sections";
    for (unsigned int i = 0; i < pgonZHalf.size(); i++)
      edm::LogVerbatim("HCalGeom") << "\t\tZ = " << convertCmToMm(pgonZHalf[i])
                                   << "\tRmin = " << convertCmToMm(pgonRminHalf[i])
                                   << "\tRmax = " << convertCmToMm(pgonRmaxHalf[i]);
#endif

    for (int ii = 0; ii < nsectortot; ii++) {
      double phi = ii * 2 * alpha;
      dd4hep::Rotation3D rotation;
      if (phi != 0) {
        rotation = dd4hep::RotationZ(phi);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Creating a new "
                                     << "rotation around Z " << convertRadToDeg(phi);
#endif
      }  //if phideg!=0
      genlogich.placeVolume(seclogic, ii + 1, rotation);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << seclogic.name() << " number " << ii + 1
                                   << " positioned in " << genlogich.name() << " at (0, 0, 0) with " << rotation;
#endif
    }

    //Construct the things inside the sector
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: Layers (" << nLayers << ") ...";
#endif
    rin = rinner;
    for (int i = 0; i < nLayers; i++) {
      std::string name = idName + layerLabel[i];
      dd4hep::Material matter = ns.material(layerMat[i]);
      double width = layerWidth[i];
      double rout = rin + width;

      int in = 0, out = 0;
      for (int j = 0; j < rzones - 1; j++) {
        if (rin >= rmax[j])
          in = j + 1;
        if (rout > rmax[j])
          out = j + 1;
      }
      double zout = zoff[in] + rin * ttheta[in];

      //!!!!!!!!!!!!!!!!!Should be zero. And removed as soon as
      //vertical walls are allowed in SolidPolyhedra
      int nsec = 2;
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
          pgonZ.emplace_back(zoff[in] + rout * ttheta[in]);
          pgonRmin.emplace_back(pgonRmax[1]);
          pgonRmax.emplace_back(pgonRmax[1]);
          nsec++;
        }
      } else {
        if (in == 3) {
          //redo index 1, add index 2
          pgonZ[1] = (zoff[out] + rmax[out] * ttheta[out]);
          pgonZ.emplace_back(pgonZ[1]);
          pgonRmin.emplace_back(pgonRmin[1]);
          pgonRmax.emplace_back(rmax[in]);
          //index 3
          pgonZ.emplace_back(zoff[in] + rmax[in] * ttheta[in]);
          pgonRmin.emplace_back(pgonRmin[2]);
          pgonRmax.emplace_back(pgonRmax[2]);
          nsec += 2;
        } else {
          //index 2
          pgonZ.emplace_back(zoff[in] + rmax[in] * ttheta[in]);
          pgonRmin.emplace_back(rmax[in]);
          pgonRmax.emplace_back(pgonRmax[1]);
          nsec++;
          if (in == 0) {
            pgonZ.emplace_back(zoff[out] + rmax[in] * ttheta[out]);
            pgonRmin.emplace_back(pgonRmin[2]);
            pgonRmax.emplace_back(pgonRmax[2]);
            nsec++;
          }
          if (in <= 1) {
            pgonZ.emplace_back(zoff[out] + rout * ttheta[out]);
            pgonRmin.emplace_back(rout);
            pgonRmax.emplace_back(rout);
            nsec++;
          }
        }
      }
      //Solid & volume
      dd4hep::Solid solid;
      double alpha1 = alpha;
      if (layerGap[i] > 1.e-6) {
        double rmid = 0.5 * (rin + rout);
        double width = rmid * tan(alpha) - layerGap[i];
        alpha1 = atan(width / rmid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "\tAlpha_1 modified from " << convertRadToDeg(alpha) << " to "
                                     << convertRadToDeg(alpha1) << " Rmid " << convertCmToMm(rmid) << " Reduced width "
                                     << convertCmToMm(width);
#endif
      }
      solid = dd4hep::Polyhedra(ns.prepend(name), 1, -alpha1, 2 * alpha1, pgonZ, pgonRmin, pgonRmax);
      dd4hep::Volume glog(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " (Layer " << i << ") Polyhedra made of "
                                   << matter.name() << " with 1 sector from " << convertRadToDeg(-alpha1) << " to "
                                   << convertRadToDeg(alpha1) << " and with " << nsec << " sections";
      for (unsigned int k = 0; k < pgonZ.size(); k++)
        edm::LogVerbatim("HCalGeom") << "\t\t" << convertCmToMm(pgonZ[k]) << "\t" << convertCmToMm(pgonRmin[k]) << "\t"
                                     << convertCmToMm(pgonRmax[k]);
#endif

      seclogic.placeVolume(glog, layerId[i]);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " number " << layerId[i]
                                   << " positioned in " << seclogic.name() << " at (0,0,0) with no rotation";
#endif
      constructInsideLayers(glog,
                            layerLabel[i],
                            layerId[i],
                            layerAbsorb[i],
                            rin,
                            layerD1[i],
                            alpha1,
                            layerD2[i],
                            layerAlpha[i],
                            layerT1[i],
                            layerT2[i],
                            ns);
      rin = rout;
    }
  }

  void constructInsideLayers(dd4hep::Volume& laylog,
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
                             cms::DDNamespace& ns) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: \t\tInside layer " << id << "...";
#endif

    ///////////////////////////////////////////////////////////////
    //Pointers to the Rotation Matrices and to the Materials
    dd4hep::Rotation3D rot = getRotation(detRot, rotns, ns);

    std::string nam0 = nm + "In";
    std::string name = idName + nam0;
    dd4hep::Material matter = ns.material(detMat);

    dd4hep::Solid solid;
    dd4hep::Volume glog, mother;
    double rsi, dx, dy, dz, x, y;
    int i, in;
    //Two lower volumes
    if (alpha1 > 0) {
      rsi = rin + d1;
      in = 0;
      for (i = 0; i < rzones - 1; i++) {
        if (rsi >= rmax[i])
          in = i + 1;
      }
      dx = 0.5 * t1;
      dy = 0.5 * rsi * (tan(alpha1) - tan(alpha2));
      dz = 0.5 * (zoff[in] + rsi * ttheta[in]);
      x = rsi + dx;
      y = 0.5 * rsi * (tan(alpha1) + tan(alpha2));
      dd4hep::Position r11(x, y, dz);
      dd4hep::Position r12(x, -y, dz);

      solid = dd4hep::Box(ns.prepend(name + "1"), dx, dy, dz);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Box made of " << matter.name()
                                   << " of dimensions " << convertCmToMm(dx) << ", " << convertCmToMm(dy) << ", "
                                   << convertCmToMm(dz);
#endif
      glog = dd4hep::Volume(solid.name(), solid, matter);

      if (nAbs != 0) {
        mother = constructSideLayer(laylog, name, nAbs, rin, alpha1, ns);
      } else {
        mother = laylog;
      }
      mother.placeVolume(glog, idOffset + 1, r11);
      mother.placeVolume(glog, idOffset + 2, dd4hep::Transform3D(rot, r12));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " Number " << idOffset + 1
                                   << " positioned in " << mother.name() << " at (" << convertCmToMm(x) << ","
                                   << convertCmToMm(y) << "," << convertCmToMm(dz) << ") with no rotation\n"
                                   << "DDHCalBarrelAlgo: " << glog.name() << " Number " << idOffset + 2
                                   << " positioned in " << mother.name() << " at (" << convertCmToMm(x) << ","
                                   << -convertCmToMm(y) << "," << convertCmToMm(dz) << ") with " << rot;
#endif
      //Constructin the plastics and scintillators inside
      constructInsideDetectors(glog, nam0 + "1", id, dx, dy, dz, 1, ns);
    }

    //Upper volume
    rsi = rin + d2;
    in = 0;
    for (i = 0; i < rzones - 1; i++) {
      if (rsi >= rmax[i])
        in = i + 1;
    }
    dx = 0.5 * t2;
    dy = 0.5 * rsi * tan(alpha2);
    dz = 0.5 * (zoff[in] + rsi * ttheta[in]);
    x = rsi + dx;
    dd4hep::Position r21(x, dy, dz);
    dd4hep::Position r22(x, -dy, dz);

    solid = dd4hep::Box(ns.prepend(name + "2"), dx, dy, dz);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Box made of " << matter.name()
                                 << " of dimensions " << convertCmToMm(dx) << ", " << convertCmToMm(dy) << ", "
                                 << convertCmToMm(dz);
#endif
    glog = dd4hep::Volume(solid.name(), solid, matter);

    if (nAbs < 0) {
      mother = constructMidLayer(laylog, name, rin, alpha1, ns);
    } else {
      mother = laylog;
    }
    mother.placeVolume(glog, idOffset + 3, r21);
    mother.placeVolume(glog, idOffset + 4, dd4hep::Transform3D(rot, r22));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " Number " << idOffset + 3
                                 << " positioned in " << mother.name() << " at (" << convertCmToMm(x) << ","
                                 << convertCmToMm(dy) << "," << convertCmToMm(dz)
                                 << ") with no rotation\nDDHCalBarrelAlgo: " << glog.name() << " Number "
                                 << idOffset + 4 << " positioned in " << mother.name() << " at (" << convertCmToMm(x)
                                 << "," << -convertCmToMm(dy) << "," << convertCmToMm(dz) << ") with " << rot;
#endif
    //Constructin the plastics and scintillators inside
    constructInsideDetectors(glog, nam0 + "2", id, dx, dy, dz, 2, ns);
  }

  dd4hep::Volume constructSideLayer(
      dd4hep::Volume& laylog, const std::string& nm, int nAbs, double rin, double alpha, cms::DDNamespace& ns) {
    //Extra absorber layer
    int k = abs(nAbs) - 1;
    std::string namek = nm + "Side";
    double rsi = rin + sideD[k];
    int in = 0;
    for (int i = 0; i < rzones - 1; i++) {
      if (rsi >= rmax[i])
        in = i + 1;
    }
    std::vector<double> pgonZ, pgonRmin, pgonRmax;
    // index 0
    pgonZ.emplace_back(0.0);
    pgonRmin.emplace_back(rsi);
    pgonRmax.emplace_back(rsi + sideT[k]);
    // index 1
    pgonZ.emplace_back(zoff[in] + rsi * ttheta[in]);
    pgonRmin.emplace_back(rsi);
    pgonRmax.emplace_back(pgonRmax[0]);
    // index 2
    pgonZ.emplace_back(zoff[in] + pgonRmax[0] * ttheta[in]);
    pgonRmin.emplace_back(pgonRmax[1]);
    pgonRmax.emplace_back(pgonRmax[1]);
    dd4hep::Solid solid = dd4hep::Polyhedra(ns.prepend(namek), 1, -alpha, 2 * alpha, pgonZ, pgonRmin, pgonRmax);
    dd4hep::Material matter = ns.material(sideMat[k]);
    dd4hep::Volume glog(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of " << sideMat[k]
                                 << " with 1 sector from " << convertRadToDeg(-alpha) << " to "
                                 << convertRadToDeg(alpha) << " and with " << pgonZ.size() << " sections";
    for (unsigned int ii = 0; ii < pgonZ.size(); ii++)
      edm::LogVerbatim("HCalGeom") << "\t\tZ = " << convertCmToMm(pgonZ[ii])
                                   << "\tRmin = " << convertCmToMm(pgonRmin[ii])
                                   << "\tRmax = " << convertCmToMm(pgonRmax[ii]);
#endif

    laylog.placeVolume(glog, 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " Number 1 positioned in " << laylog.name()
                                 << " at (0,0,0) with no rotation";
#endif
    if (nAbs < 0) {
      dd4hep::Volume mother = glog;
      double rmid = pgonRmax[0];
      for (int i = 0; i < nSideAbs; i++) {
        double alpha1 = atan(sideAbsW[i] / rmid);
        if (alpha1 > 0) {
          std::string name = namek + sideAbsName[i];
          solid = dd4hep::Polyhedra(ns.prepend(name), 1, -alpha1, 2 * alpha1, pgonZ, pgonRmin, pgonRmax);
          dd4hep::Material matter = ns.material(sideAbsMat[i]);
          dd4hep::Volume log(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of " << sideAbsMat[i]
                                       << " with 1 sector from " << convertRadToDeg(-alpha1) << " to "
                                       << convertRadToDeg(alpha1) << " and with " << pgonZ.size() << " sections";
          for (unsigned int ii = 0; ii < pgonZ.size(); ii++)
            edm::LogVerbatim("HCalGeom") << "\t\tZ = " << convertCmToMm(pgonZ[ii])
                                         << "\tRmin = " << convertCmToMm(pgonRmin[ii])
                                         << "\tRmax = " << convertCmToMm(pgonRmax[ii]);
#endif

          mother.placeVolume(log, 1);
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

  dd4hep::Volume constructMidLayer(
      dd4hep::Volume laylog, const std::string& nm, double rin, double alpha, cms::DDNamespace& ns) {
    dd4hep::Solid solid;
    dd4hep::Volume log, glog;
    std::string name = nm + "Mid";
    for (int k = 0; k < nAbsorber; k++) {
      std::string namek = name + absorbName[k];
      double rsi = rin + absorbD[k];
      int in = 0;
      for (int i = 0; i < rzones - 1; i++) {
        if (rsi >= rmax[i])
          in = i + 1;
      }
      std::vector<double> pgonZ, pgonRmin, pgonRmax;
      // index 0
      pgonZ.emplace_back(0.0);
      pgonRmin.emplace_back(rsi);
      pgonRmax.emplace_back(rsi + absorbT[k]);
      // index 1
      pgonZ.emplace_back(zoff[in] + rsi * ttheta[in]);
      pgonRmin.emplace_back(rsi);
      pgonRmax.emplace_back(pgonRmax[0]);
      // index 2
      pgonZ.emplace_back(zoff[in] + pgonRmax[0] * ttheta[in]);
      pgonRmin.emplace_back(pgonRmax[1]);
      pgonRmax.emplace_back(pgonRmax[1]);
      solid = dd4hep::Polyhedra(ns.prepend(namek), 1, -alpha, 2 * alpha, pgonZ, pgonRmin, pgonRmax);
      dd4hep::Material matter = ns.material(absorbMat[k]);
      log = dd4hep::Volume(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of " << matter.name()
                                   << " with 1 sector from " << convertRadToDeg(-alpha) << " to "
                                   << convertRadToDeg(alpha) << " and with " << pgonZ.size() << " sections";
      for (unsigned int ii = 0; ii < pgonZ.size(); ii++)
        edm::LogVerbatim("HCalGeom") << "\t\tZ = " << convertCmToMm(pgonZ[ii])
                                     << "\tRmin = " << convertCmToMm(pgonRmin[ii])
                                     << "\tRmax = " << convertCmToMm(pgonRmax[ii]);
#endif

      laylog.placeVolume(log, 1);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << log.name() << " Number 1 positioned in " << laylog.name()
                                   << " at (0,0,0) with no rotation";
#endif
      if (k == 0) {
        double rmin = pgonRmin[0];
        double rmax = pgonRmax[0];
        dd4hep::Volume mother = log;
        for (int i = 0; i < 1; i++) {
          double alpha1 = atan(midW[i] / rmin);
          std::string namek = name + midName[i];
          solid = dd4hep::Polyhedra(ns.prepend(namek), 1, -alpha1, 2 * alpha1, pgonZ, pgonRmin, pgonRmax);
          dd4hep::Material matter1 = ns.material(midMat[i]);
          log = dd4hep::Volume(solid.name(), solid, matter1);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of "
                                       << matter1.name() << " with 1 sector from " << convertRadToDeg(-alpha1) << " to "
                                       << convertRadToDeg(alpha1) << " and with " << pgonZ.size() << " sections";
          for (unsigned int ii = 0; ii < pgonZ.size(); ii++)
            edm::LogVerbatim("HCalGeom") << "\t\tZ = " << convertCmToMm(pgonZ[ii])
                                         << "\tRmin = " << convertCmToMm(pgonRmin[ii])
                                         << "\tRmax = " << convertCmToMm(pgonRmax[ii]);
#endif

          mother.placeVolume(log, 1);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << log.name() << " Number 1 positioned in "
                                       << mother.name() << " at (0,0,0) with no rotation";
#endif
          mother = log;
        }

        // Now the layer with detectors
        double rmid = rmin + middleD;
        pgonRmin[0] = rmid;
        pgonRmax[0] = rmax;
        pgonRmin[1] = rmid;
        pgonRmax[1] = rmax;
        pgonZ[1] = zoff[in] + rmid * ttheta[in];
        pgonRmin[2] = rmax;
        pgonRmax[2] = rmax;
        pgonZ[2] = zoff[in] + rmax * ttheta[in];
        double alpha1 = atan(middleW / rmin);
        solid = dd4hep::Polyhedra(ns.prepend(name), 1, -alpha1, 2 * alpha1, pgonZ, pgonRmin, pgonRmax);
        dd4hep::Material matter1 = ns.material(middleMat);
        glog = dd4hep::Volume(solid.name(), solid, matter1);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of " << matter1.name()
                                     << " with 1 sector from " << convertRadToDeg(-alpha1) << " to "
                                     << convertRadToDeg(alpha1) << " and with " << pgonZ.size() << " sections";
        for (unsigned int ii = 0; ii < pgonZ.size(); ii++)
          edm::LogVerbatim("HCalGeom") << "\t\tZ = " << convertCmToMm(pgonZ[ii])
                                       << "\tRmin = " << convertCmToMm(pgonRmin[ii])
                                       << "\tRmax = " << convertCmToMm(pgonRmax[ii]);
#endif

        mother.placeVolume(glog, 1);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " Number 1 positioned in "
                                     << mother.name() << " at (0,0,0) with no rotation";
#endif
        // Now the remaining absorber layers
        for (int i = 1; i < nMidAbs; i++) {
          namek = name + midName[i];
          rmid = rmin + midT[i];
          pgonRmin[0] = rmin;
          pgonRmax[0] = rmid;
          pgonRmin[1] = rmin;
          pgonRmax[1] = rmid;
          pgonZ[1] = zoff[in] + rmin * ttheta[in];
          pgonRmin[2] = rmid;
          pgonRmax[2] = rmid;
          pgonZ[2] = zoff[in] + rmid * ttheta[in];
          alpha1 = atan(midW[i] / rmin);
          solid = dd4hep::Polyhedra(ns.prepend(namek), 1, -alpha1, 2 * alpha1, pgonZ, pgonRmin, pgonRmax);
          dd4hep::Material matter2 = ns.material(midMat[i]);
          log = dd4hep::Volume(solid.name(), solid, matter2);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Polyhedra made of "
                                       << matter2.name() << " with 1 sector from " << convertRadToDeg(-alpha1) << " to "
                                       << convertRadToDeg(alpha1) << " and with " << pgonZ.size() << " sections";
          for (unsigned int ii = 0; ii < pgonZ.size(); ii++)
            edm::LogVerbatim("HCalGeom") << "\t\tZ = " << convertCmToMm(pgonZ[ii])
                                         << "\tRmin = " << convertCmToMm(pgonRmin[ii])
                                         << "\tRmax = " << convertCmToMm(pgonRmax[ii]);
#endif

          mother.placeVolume(log, i);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << log.name() << " Number " << i << " positioned in "
                                       << mother.name() << " at (0,0,0) with no rotation";
#endif
          mother = log;
        }
      }
    }
    return glog;
  }

  void constructInsideDetectors(dd4hep::Volume& detector,
                                const std::string& name,
                                int id,
                                double dx,
                                double dy,
                                double dz,
                                int type,
                                cms::DDNamespace& ns) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: \t\tInside detector " << id << "...";
#endif

    dd4hep::Material plmatter = ns.material(detMatPl);
    dd4hep::Material scmatter = ns.material(detMatSc);
    std::string plname = DDSplit(detector.name()).first + "Plastic_";
    std::string scname = idName + "Scintillator" + name;

    id--;
    dd4hep::Solid solid;
    dd4hep::Volume glog;
    double wid, y = 0;
    double dx1, dx2, shiftX;

    if (type == 1) {
      wid = 0.5 * detWidth1[id];
      dx1 = 0.5 * detT11[id];
      dx2 = 0.5 * detT21[id];
      shiftX = detdP1[id];
      if (detPosY[id] > 0)
        y = -dy + wid;
    } else {
      wid = 0.5 * detWidth2[id];
      dx1 = 0.5 * detT12[id];
      dx2 = 0.5 * detT22[id];
      shiftX = detdP2[id];
    }

    solid = dd4hep::Box(ns.prepend(plname + "1"), dx1, wid, dz);
    glog = dd4hep::Volume(solid.name(), solid, plmatter);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Box made of " << plmatter.name()
                                 << " of dimensions " << convertCmToMm(dx1) << ", " << convertCmToMm(wid) << ", "
                                 << convertCmToMm(dz);
#endif

    double x = shiftX + dx1 - dx;
    detector.placeVolume(glog, 1, dd4hep::Position(x, y, 0));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " Number 1 positioned in " << detector.name()
                                 << " at (" << convertCmToMm(x) << "," << convertCmToMm(y) << ",0) with no rotation";
#endif
    solid = dd4hep::Box(ns.prepend(scname), 0.5 * detTsc[id], wid, dz);
    glog = dd4hep::Volume(solid.name(), solid, scmatter);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Box made of " << scmatter.name()
                                 << " of dimensions " << convertCmToMm(0.5 * detTsc[id]) << ", " << convertCmToMm(wid)
                                 << ", " << convertCmToMm(dz);
#endif

    x += dx1 + 0.5 * detTsc[id];
    int copyNo = id * 10 + detType[id];
    detector.placeVolume(glog, copyNo, dd4hep::Position(x, y, 0));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " Number " << copyNo << " positioned in "
                                 << detector.name() << " at (" << convertCmToMm(x) << "," << convertCmToMm(y)
                                 << ",0) with no rotation";
#endif
    solid = dd4hep::Box(ns.prepend(plname + "2"), dx2, wid, dz);
    glog = dd4hep::Volume(solid.name(), solid, plmatter);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << solid.name() << " Box made of " << plmatter.name()
                                 << " of dimensions " << convertCmToMm(dx2) << ", " << convertCmToMm(wid) << ", "
                                 << convertCmToMm(dz);
#endif

    x += 0.5 * detTsc[id] + dx2;
    detector.placeVolume(glog, 1, dd4hep::Position(x, y, 0));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalBarrelAlgo: " << glog.name() << " Number 1 positioned in " << detector.name()
                                 << " at (" << convertCmToMm(x) << "," << convertCmToMm(y) << ",0) with no rotation";
#endif
  }

  dd4hep::Rotation3D getRotation(std::string& rotation, std::string& rotns, cms::DDNamespace& ns) {
    std::string rot = (strchr(rotation.c_str(), NAMESPACE_SEP) == nullptr) ? (rotns + ":" + rotation) : rotation;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "getRotation: " << rotation << ":" << rot << ":" << ns.rotation(rot);
#endif
    return ns.rotation(rot);
  }
};

static long algorithm(dd4hep::Detector& /* description */,
                      cms::DDParsingContext& ctxt,
                      xml_h e,
                      dd4hep::SensitiveDetector& /* sens */) {
  HcalBarrelAlgo hcalbarrelalgo(ctxt, e);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "<<== End of DDHCalBarrelAlgo construction";
#endif
  return cms::s_executed;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hcal_DDHCalBarrelAlgo, algorithm)
