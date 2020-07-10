///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalEEFileAlgo.cc
// Description: Geometry factory class for HGCal (EE and HESil) using
//              information from the file for dd4hep
///////////////////////////////////////////////////////////////////////////////
#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"
#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG
using namespace cms_units::operators;

struct HGCalEEFileAlgo {
  HGCalEEFileAlgo() { throw cms::Exception("HGCalGeom") << "Wrong initialization to HGCalEEFileAlgo"; }
  HGCalEEFileAlgo(cms::DDParsingContext& ctxt, xml_h e) {
    cms::DDNamespace ns(ctxt, e, true);
    cms::DDAlgoArguments args(ctxt, e);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: Creating an instance";
#endif
    static constexpr double tol1 = 0.01;
    static constexpr double tol2 = 0.00001;

    dd4hep::Volume mother = ns.volume(args.parentName());
    wafers = args.value<std::vector<std::string>>("WaferNames");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << wafers.size() << " wafers";
    for (unsigned int i = 0; i < wafers.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << "] " << wafers[i];
#endif
    materials = args.value<std::vector<std::string>>("MaterialNames");
    names = args.value<std::vector<std::string>>("VolumeNames");
    thick = args.value<std::vector<double>>("Thickness");
    copyNumber.resize(materials.size(), 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << materials.size() << " types of volumes";
    for (unsigned int i = 0; i < names.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << names[i] << " of thickness " << thick[i]
                                    << " filled with " << materials[i] << " first copy number " << copyNumber[i];
#endif
    layers = args.value<std::vector<int>>("Layers");
    layerThick = args.value<std::vector<double>>("LayerThick");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "There are " << layers.size() << " blocks";
    for (unsigned int i = 0; i < layers.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << layerThick[i] << " with " << layers[i]
                                    << " layers";
#endif
    layerType = args.value<std::vector<int>>("LayerType");
    layerSense = args.value<std::vector<int>>("LayerSense");
    firstLayer = args.value<int>("FirstLayer");
    absorbMode = args.value<int>("AbsorberMode");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "First Layer " << firstLayer << " and "
                                  << "Absober mode " << absorbMode;
#endif
    layerCenter = args.value<std::vector<int>>("LayerCenter");
#ifdef EDM_ML_DEBUG
    for (unsigned int i = 0; i < layerCenter.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "LayerCenter [" << i << "] " << layerCenter[i];
#endif
    if (firstLayer > 0) {
      for (unsigned int i = 0; i < layerType.size(); ++i) {
        if (layerSense[i] > 0) {
          int ii = layerType[i];
          copyNumber[ii] = firstLayer;
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "First copy number for layer type " << i << ":" << ii << " with "
                                        << materials[ii] << " changed to " << copyNumber[ii];
#endif
          break;
        }
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "There are " << layerType.size() << " layers";
    for (unsigned int i = 0; i < layerType.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerType[i] << " sensitive class "
                                    << layerSense[i];
#endif
    zMinBlock = args.value<double>("zMinBlock");
    rad100to200 = args.value<std::vector<double>>("rad100to200");
    rad200to300 = args.value<std::vector<double>>("rad200to300");
    zMinRadPar = args.value<double>("zMinForRadPar");
    choiceType = args.value<int>("choiceType");
    nCutRadPar = args.value<int>("nCornerCut");
    fracAreaMin = args.value<double>("fracAreaMin");
    waferSize = args.value<double>("waferSize");
    waferSepar = args.value<double>("SensorSeparation");
    sectors = args.value<int>("Sectors");
    alpha = (1._pi) / sectors;
    cosAlpha = cos(alpha);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "zStart " << zMinBlock << " radius for wafer type separation uses "
                                  << rad100to200.size() << " parameters; zmin " << zMinRadPar << " cutoff "
                                  << choiceType << ":" << nCutRadPar << ":" << fracAreaMin << " wafer width "
                                  << waferSize << " separations " << waferSepar << " sectors " << sectors << ":"
                                  << convertRadToDeg(alpha) << ":" << cosAlpha;
    for (unsigned int k = 0; k < rad100to200.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] 100-200 " << rad100to200[k] << " 200-300 " << rad200to300[k];
#endif
    waferIndex = args.value<std::vector<int>>("WaferIndex");
    waferTypes = args.value<std::vector<int>>("WaferTypes");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "waferTypes with " << waferTypes.size() << " entries";
    for (unsigned int k = 0; k < waferTypes.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << waferIndex[k] << " ("
                                    << HGCalWaferIndex::waferLayer(waferIndex[k]) << ", "
                                    << HGCalWaferIndex::waferU(waferIndex[k]) << ", "
                                    << HGCalWaferIndex::waferV(waferIndex[k]) << ") : " << waferTypes[k];
#endif
    slopeB = args.value<std::vector<double>>("SlopeBottom");
    zFrontB = args.value<std::vector<double>>("ZFrontBottom");
    rMinFront = args.value<std::vector<double>>("RMinFront");
    slopeT = args.value<std::vector<double>>("SlopeTop");
    zFrontT = args.value<std::vector<double>>("ZFrontTop");
    rMaxFront = args.value<std::vector<double>>("RMaxFront");
#ifdef EDM_ML_DEBUG
    for (unsigned int i = 0; i < slopeB.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] Zmin " << zFrontB[i] << " Rmin " << rMinFront[i]
                                    << " Slope " << slopeB[i];
    for (unsigned int i = 0; i < slopeT.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] Zmin " << zFrontT[i] << " Rmax " << rMaxFront[i]
                                    << " Slope " << slopeT[i];
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: NameSpace " << ns.name();
#endif

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalEEFileAlgo...";
    copies.clear();
#endif

    double zi(zMinBlock);
    int laymin(0);
    for (unsigned int i = 0; i < layers.size(); ++i) {
      double zo = zi + layerThick[i];
      double routF = HGCalGeomTools::radius(zi, zFrontT, rMaxFront, slopeT);
      int laymax = laymin + layers[i];
      double zz = zi;
      double thickTot(0);
      for (int ly = laymin; ly < laymax; ++ly) {
        int ii = layerType[ly];
        int copy = copyNumber[ii];
        double hthick = 0.5 * thick[ii];
        double rinB = HGCalGeomTools::radius(zo, zFrontB, rMinFront, slopeB);
        zz += hthick;
        thickTot += thick[ii];

        std::string name = names[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: Layer " << ly << ":" << ii << " Front " << zi << ", "
                                      << routF << " Back " << zo << ", " << rinB << " superlayer thickness "
                                      << layerThick[i];
#endif

        dd4hep::Material matter = ns.material(materials[ii]);
        dd4hep::Volume glog;

        if (layerSense[ly] < 1) {
          std::vector<double> pgonZ, pgonRin, pgonRout;
          if (layerSense[ly] == 0 || absorbMode == 0) {
            double rmax = routF * cosAlpha - tol1;
            pgonZ.emplace_back(-hthick);
            pgonZ.emplace_back(hthick);
            pgonRin.emplace_back(rinB);
            pgonRin.emplace_back(rinB);
            pgonRout.emplace_back(rmax);
            pgonRout.emplace_back(rmax);
          } else {
            HGCalGeomTools::radius(zz - hthick,
                                   zz + hthick,
                                   zFrontB,
                                   rMinFront,
                                   slopeB,
                                   zFrontT,
                                   rMaxFront,
                                   slopeT,
                                   -layerSense[ly],
                                   pgonZ,
                                   pgonRin,
                                   pgonRout);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: z " << (zz - hthick) << ":" << (zz + hthick)
                                          << " with " << pgonZ.size() << " palnes";
            for (unsigned int isec = 0; isec < pgonZ.size(); ++isec)
              edm::LogVerbatim("HGCalGeom")
                  << "[" << isec << "] z " << pgonZ[isec] << " R " << pgonRin[isec] << ":" << pgonRout[isec];
#endif
            for (unsigned int isec = 0; isec < pgonZ.size(); ++isec) {
              pgonZ[isec] -= zz;
              pgonRout[isec] = pgonRout[isec] * cosAlpha - tol1;
            }
          }
          dd4hep::Solid solid = dd4hep::Polyhedra(sectors, -alpha, 2._pi, pgonZ, pgonRin, pgonRout);
          ns.addSolidNS(ns.prepend(name), solid);
          glog = dd4hep::Volume(solid.name(), solid, matter);
          ns.addVolumeNS(glog);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << solid.name() << " polyhedra of " << sectors
                                        << " sectors covering " << convertRadToDeg(-alpha) << ":"
                                        << convertRadToDeg(-alpha + 2._pi) << " with " << pgonZ.size()
                                        << " sections and filled with " << matter.name();
          for (unsigned int k = 0; k < pgonZ.size(); ++k)
            edm::LogVerbatim("HGCalGeom")
                << "[" << k << "] z " << pgonZ[k] << " R " << pgonRin[k] << ":" << pgonRout[k];
#endif
        } else {
          dd4hep::Solid solid = dd4hep::Tube(rinB, routF, hthick, 0.0, 2._pi);
          ns.addSolidNS(ns.prepend(name), solid);
          glog = dd4hep::Volume(solid.name(), solid, matter);
          ns.addVolumeNS(glog);

#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << solid.name() << " Tubs made of " << matter.name()
                                        << " of dimensions " << rinB << ", " << routF << ", " << hthick
                                        << ", 0.0, 360.0 and position " << glog.name() << " number " << copy << ":"
                                        << layerCenter[copy - 1];
#endif
          positionSensitive(ctxt, e, glog, rinB, routF, zz, layerSense[ly], (copy - 1));
        }

        dd4hep::Position r1(0, 0, zz);
        mother.placeVolume(glog, copy, r1);
        ++copyNumber[ii];
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << glog.name() << " number " << copy << " positioned in "
                                      << mother.name() << " at " << r1 << " with no rotation";
#endif
        zz += hthick;
      }  // End of loop over layers in a block
      zi = zo;
      laymin = laymax;
      // Make consistency check of all the partitions of the block
      if (std::abs(thickTot - layerThick[i]) >= tol2) {
        if (thickTot > layerThick[i]) {
          edm::LogError("HGCalGeom") << "Thickness of the partition " << layerThick[i] << " is smaller than "
                                     << thickTot << ": thickness of all its components **** ERROR ****";
        } else {
          edm::LogWarning("HGCalGeom") << "Thickness of the partition " << layerThick[i] << " does not match with "
                                       << thickTot << " of the components";
        }
      }
    }  // End of loop over blocks

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << copies.size() << " different wafer copy numbers";
    int k(0);
    for (std::unordered_set<int>::const_iterator itr = copies.begin(); itr != copies.end(); ++itr, ++k) {
      edm::LogVerbatim("HGCalGeom") << "Copy [" << k << "] : " << (*itr);
    }
    copies.clear();
    edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalEEFileAlgo construction...";
#endif
  }

  void positionSensitive(cms::DDParsingContext& ctxt,
                         xml_h e,
                         const dd4hep::Volume& glog,
                         double rin,
                         double rout,
                         double zpos,
                         int layertype,
                         int layer) {
    cms::DDNamespace ns(ctxt, e, true);
    static const double sqrt3 = std::sqrt(3.0);
    int layercenter = layerCenter[layer];
    double r = 0.5 * (waferSize + waferSepar);
    double R = 2.0 * r / sqrt3;
    double dy = 0.75 * R;
    int N = (int)(0.5 * rout / r) + 2;
    const auto& xyoff = geomTools.shiftXY(layercenter, (waferSize + waferSepar));
#ifdef EDM_ML_DEBUG
    int ium(0), ivm(0), iumAll(0), ivmAll(0), kount(0), ntot(0), nin(0);
    std::vector<int> ntype(6, 0);
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << glog.name() << " rin:rout " << rin << ":" << rout
                                  << " zpos " << zpos << " N " << N << " for maximum u, v;  r " << r << " R " << R
                                  << " dy " << dy << " Shift " << xyoff.first << ":" << xyoff.second << " WaferSize "
                                  << (waferSize + waferSepar);
#endif
    for (int u = -N; u <= N; ++u) {
      for (int v = -N; v <= N; ++v) {
#ifdef EDM_ML_DEBUG
	int iu = std::abs(u);
        int iv = std::abs(v);
#endif
        int nr = 2 * v;
        int nc = -2 * u + v;
        double xpos = xyoff.first + nc * r;
        double ypos = xyoff.second + nr * dy;
        const auto& corner = HGCalGeomTools::waferCorner(xpos, ypos, r, R, rin, rout, false);
#ifdef EDM_ML_DEBUG
        ++ntot;
        if (((corner.first <= 0) && std::abs(u) < 5 && std::abs(v) < 5) || (std::abs(u) < 2 && std::abs(v) < 2)) {
          edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << glog.name() << " R " << rin << ":" << rout
                                        << "\n Z " << zpos << " LayerType " << layertype << " u " << u << " v " << v
                                        << " with " << corner.first << " corners";
        }
#endif
        int type = HGCalWaferType::getType(HGCalWaferIndex::waferIndex(layer, u, v, false), waferIndex, waferTypes);
        if (corner.first > 0 && type >= 0) {
          int copy = HGCalTypes::packTypeUV (type, u, v);
#ifdef EDM_ML_DEBUG
          if (iu > ium)
            ium = iu;
          if (iv > ivm)
            ivm = iv;
          kount++;
          if (copies.count(copy) == 0)
            copies.insert(copy);
#endif
          if (corner.first == (int)(HGCalParameters::k_CornerSize)) {
#ifdef EDM_ML_DEBUG
            if (iu > iumAll)
              iumAll = iu;
            if (iv > ivmAll)
              ivmAll = iv;
            ++nin;
#endif
            dd4hep::Position tran(xpos, ypos, 0.0);
            if (layertype > 1)
              type += 3;
            glog.placeVolume(ns.volume(wafers[type]), copy, tran);
#ifdef EDM_ML_DEBUG
            ++ntype[type];
            edm::LogVerbatim("HGCalGeom")
                << " DDHGCalEEFileAlgo: " << wafers[type] << " number " << copy << " type " << layertype << ":" << type
                << " positioned in " << glog.name() << " at " << tran << " with no rotation";
#endif
          }
        }
      }
    }

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: Maximum # of u " << ium << ":" << iumAll << " # of v " << ivm
                                  << ":" << ivmAll << " and " << nin << ":" << kount << ":" << ntot << " wafers ("
                                  << ntype[0] << ":" << ntype[1] << ":" << ntype[2] << ":" << ntype[3] << ":"
                                  << ntype[4] << ":" << ntype[5] << ") for " << glog.name() << " R " << rin << ":"
                                  << rout;
#endif
  }

  //Required data members to cache the values from XML file
  HGCalGeomTools geomTools;

  std::vector<std::string> wafers;     // Wafers
  std::vector<std::string> materials;  // Materials
  std::vector<std::string> names;      // Names
  std::vector<double> thick;           // Thickness of the material
  std::vector<int> copyNumber;         // Initial copy numbers
  std::vector<int> layers;             // Number of layers in a section
  std::vector<double> layerThick;      // Thickness of each section
  std::vector<int> layerType;          // Type of the layer
  std::vector<int> layerSense;         // Content of a layer (sensitive?)
  std::vector<int> layerCenter;        // Centering of the wafers
  int firstLayer;                      // Copy # of the first sensitive layer
  int absorbMode;                      // Absorber mode
  double zMinBlock;                    // Starting z-value of the block
  std::vector<double> rad100to200;     // Parameters for 120-200mum trans.
  std::vector<double> rad200to300;     // Parameters for 200-300mum trans.
  double zMinRadPar;                   // Minimum z for radius parametriz.
  std::vector<int> waferIndex;         // Wafer index for the types
  std::vector<int> waferTypes;         // Wafer types
  int choiceType;                      // Type of parametrization to be used
  int nCutRadPar;                      // Cut off threshold for corners
  double fracAreaMin;                  // Minimum fractional conatined area
  double waferSize;                    // Width of the wafer
  double waferSepar;                   // Sensor separation
  int sectors;                         // Sectors
  std::vector<double> slopeB;          // Slope at the lower R
  std::vector<double> zFrontB;         // Starting Z values for the slopes
  std::vector<double> rMinFront;       // Corresponding rMin's
  std::vector<double> slopeT;          // Slopes at the larger R
  std::vector<double> zFrontT;         // Starting Z values for the slopes
  std::vector<double> rMaxFront;       // Corresponding rMax's
  std::unordered_set<int> copies;      // List of copy #'s
  double alpha, cosAlpha;
};

static long algorithm(dd4hep::Detector& /* description */,
                      cms::DDParsingContext& ctxt,
                      xml_h e,
                      dd4hep::SensitiveDetector& /* sens */) {
  HGCalEEFileAlgo eealgo(ctxt, e);
  return cms::s_executed;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalEEFileAlgo, algorithm)
