/*
 * DDHGCalModule.cc
 *
 *  Created on: 26-August-2019
 *      Author: Sunanda Banerjee
 */

#include "DataFormats/Math/interface/CMSUnits.h"
#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"

//#define EDM_ML_DEBUG

#ifdef EDM_ML_DEBUG
#include <unordered_set>
#endif
using namespace cms_units::operators;

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);
  static constexpr double tol = 0.01;
  static constexpr double tol2 = 0.00001;

  const auto& wafer = args.value<std::vector<std::string> >("WaferName");    // Wafers
  auto materials = args.value<std::vector<std::string> >("MaterialNames");   // Materials
  const auto& names = args.value<std::vector<std::string> >("VolumeNames");  // Names
  const auto& thick = args.value<std::vector<double> >("Thickness");         // Thickness of the material
  std::vector<int> copyNumber;                                               // Initial copy numbers
  copyNumber.resize(materials.size(), 1);
  for (unsigned int i = 0; i < materials.size(); ++i) {
    if (materials[i] == "materials:M_NEMAFR4plate")
      materials[i] = "materials:M_NEMA FR4 plate";
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << wafer.size() << " wafers";
  for (unsigned int i = 0; i < wafer.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << "] " << wafer[i];
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << materials.size() << " types of volumes";
  for (unsigned int i = 0; i < names.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << names[i] << " of thickness " << thick[i]
                                  << " filled with " << materials[i] << " first copy number " << copyNumber[i];
#endif
  const auto& layers = args.value<std::vector<int> >("Layers");             // Number of layers in a section
  const auto& layerThick = args.value<std::vector<double> >("LayerThick");  // Thickness of each section
  const auto& layerType = args.value<std::vector<int> >("LayerType");       // Type of the layer
  const auto& layerSense = args.value<std::vector<int> >("LayerSense");     // Content of a layer (sensitive?)
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << layers.size() << " blocks";
  for (unsigned int i = 0; i < layers.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << layerThick[i] << " with " << layers[i]
                                  << " layers";
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << layerType.size() << " layers";
  for (unsigned int i = 0; i < layerType.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerType[i] << " sensitive class "
                                  << layerSense[i];
#endif
  double zMinBlock = args.value<double>("zMinBlock");  // Starting z-value of the block
  double rMaxFine = args.value<double>("rMaxFine");    // Maximum r-value for fine wafer
  double waferW = args.value<double>("waferW");        // Width of the wafer
  int sectors = args.value<int>("Sectors");            // Sectors
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: zStart " << zMinBlock << " rFineCoarse " << rMaxFine
                                << " wafer width " << waferW << " sectors " << sectors;
#endif
  const auto& slopeB = args.value<std::vector<double> >("SlopeBottom");   // Slope at the lower R
  const auto& slopeT = args.value<std::vector<double> >("SlopeTop");      // Slopes at the larger R
  const auto& zFront = args.value<std::vector<double> >("ZFront");        // Starting Z values for the slopes
  const auto& rMaxFront = args.value<std::vector<double> >("RMaxFront");  // Corresponding rMax's
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: Bottom slopes " << slopeB[0] << ":" << slopeB[1] << " and "
                                << slopeT.size() << " slopes for top";
  for (unsigned int i = 0; i < slopeT.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] Zmin " << zFront[i] << " Rmax " << rMaxFront[i] << " Slope "
                                  << slopeT[i];
#endif
  std::string idNameSpace = static_cast<std::string>(ns.name());  // Namespace of this and ALL sub-parts
  const auto& idName = args.parentName();                         // Name of the "parent" volume.
#ifdef EDM_ML_DEBUG
  std::unordered_set<int> copies;  // List of copy #'s
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: NameSpace " << idNameSpace << " Mother " << idName;
#endif

  // Mother module
  dd4hep::Volume module = ns.volume(idName);

  double zi(zMinBlock);
  int laymin(0);
  for (unsigned int i = 0; i < layers.size(); i++) {
    double zo = zi + layerThick[i];
    double routF = HGCalGeomTools::radius(zi, zFront, rMaxFront, slopeT);
    int laymax = laymin + layers[i];
    double zz = zi;
    double thickTot(0);
    for (int ly = laymin; ly < laymax; ++ly) {
      int ii = layerType[ly];
      int copy = copyNumber[ii];
      double rinB = (layerSense[ly] == 0) ? (zo * slopeB[0]) : (zo * slopeB[1]);
      zz += (0.5 * thick[ii]);
      thickTot += thick[ii];

      std::string name = "HGCal" + names[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: Layer " << ly << ":" << ii << " Front " << zi << ", " << routF
                                    << " Back " << zo << ", " << rinB << " superlayer thickness " << layerThick[i];
#endif
      dd4hep::Material matter = ns.material(materials[ii]);
      dd4hep::Volume glog;
      if (layerSense[ly] == 0) {
        double alpha = cms_units::piRadians / sectors;
        double rmax = routF * cos(alpha) - tol;
        std::vector<double> pgonZ, pgonRin, pgonRout;
        pgonZ.emplace_back(-0.5 * thick[ii]);
        pgonZ.emplace_back(0.5 * thick[ii]);
        pgonRin.emplace_back(rinB);
        pgonRin.emplace_back(rinB);
        pgonRout.emplace_back(rmax);
        pgonRout.emplace_back(rmax);
        dd4hep::Solid solid = dd4hep::Polyhedra(sectors, -alpha, 2 * cms_units::piRadians, pgonZ, pgonRin, pgonRout);
        ns.addSolidNS(ns.prepend(name), solid);
        glog = dd4hep::Volume(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << solid.name() << " polyhedra of " << sectors
                                      << " sectors covering " << convertRadToDeg(-alpha) << ":"
                                      << (360.0 + convertRadToDeg(-alpha)) << " with " << pgonZ.size() << " sections";
        for (unsigned int k = 0; k < pgonZ.size(); ++k)
          edm::LogVerbatim("HGCalGeom") << "[" << k << "] z " << pgonZ[k] << " R " << pgonRin[k] << ":" << pgonRout[k];
#endif
      } else {
        dd4hep::Solid solid = dd4hep::Tube(0.5 * thick[ii], rinB, routF, 0.0, 2 * cms_units::piRadians);
        ns.addSolidNS(ns.prepend(name), solid);
        glog = dd4hep::Volume(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << solid.name() << " Tubs made of " << materials[ii]
                                      << " of dimensions " << rinB << ", " << routF << ", " << 0.5 * thick[ii]
                                      << ", 0.0, 360.0";
        edm::LogVerbatim("HGCalGeom") << "DDHGCalModule test position in: " << glog.name() << " number " << copy;
#endif
        double dx = 0.5 * waferW;
        double dy = 3.0 * dx * tan(30._deg);
        double rr = 2.0 * dx * tan(30._deg);
        int ncol = static_cast<int>(2.0 * routF / waferW) + 1;
        int nrow = static_cast<int>(routF / (waferW * tan(30._deg))) + 1;
#ifdef EDM_ML_DEBUG
        int incm(0), inrm(0), kount(0), ntot(0), nin(0), nfine(0), ncoarse(0);
        edm::LogVerbatim("HGCalGeom") << glog.name() << " rout " << routF << " Row " << nrow << " Column " << ncol;
#endif
        for (int nr = -nrow; nr <= nrow; ++nr) {
          int inr = (nr >= 0) ? nr : -nr;
          for (int nc = -ncol; nc <= ncol; ++nc) {
            int inc = (nc >= 0) ? nc : -nc;
            if (inr % 2 == inc % 2) {
              double xpos = nc * dx;
              double ypos = nr * dy;
              std::pair<int, int> corner = HGCalGeomTools::waferCorner(xpos, ypos, dx, rr, rinB, routF, true);
#ifdef EDM_ML_DEBUG
              ++ntot;
#endif
              if (corner.first > 0) {
                int copyL = HGCalTypes::packTypeUV(0, nc, nr);
#ifdef EDM_ML_DEBUG
                if (inc > incm)
                  incm = inc;
                if (inr > inrm)
                  inrm = inr;
                kount++;
                copies.insert(copy);
#endif
                if (corner.first == (int)(HGCalParameters::k_CornerSize)) {
                  double rpos = std::sqrt(xpos * xpos + ypos * ypos);
                  dd4hep::Position tran(xpos, ypos, 0.0);
                  dd4hep::Rotation3D rotation;
                  dd4hep::Volume glog1 = (rpos < rMaxFine) ? ns.volume(wafer[0]) : ns.volume(wafer[1]);
                  glog.placeVolume(glog1, copyL, dd4hep::Transform3D(rotation, tran));
#ifdef EDM_ML_DEBUG
                  ++nin;
                  if (rpos < rMaxFine)
                    ++nfine;
                  else
                    ++ncoarse;
                  edm::LogVerbatim("HGCalGeom")
                      << "DDHGCalModule: " << glog1.name() << " number " << copyL << " positioned in " << glog.name()
                      << " at " << tran << " with " << rotation;
#endif
                }
              }
            }
          }
        }
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: # of columns " << incm << " # of rows " << inrm << " and "
                                      << nin << ":" << kount << ":" << ntot << " wafers (" << nfine << ":" << ncoarse
                                      << ") for " << glog.name() << " R " << rinB << ":" << routF;
#endif
      }
      dd4hep::Position r1(0, 0, zz);
      dd4hep::Rotation3D rot;
      module.placeVolume(glog, copy, dd4hep::Transform3D(rot, r1));
      ++copyNumber[ii];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: " << glog.name() << " number " << copy << " positioned in "
                                    << module.name() << " at " << r1 << " with " << rot;
#endif
      zz += (0.5 * thick[ii]);
    }  // End of loop over layers in a block
    zi = zo;
    laymin = laymax;
    if (fabs(thickTot - layerThick[i]) > tol2) {
      if (thickTot > layerThick[i]) {
        edm::LogError("HGCalGeom") << "Thickness of the partition " << layerThick[i] << " is smaller than thickness "
                                   << thickTot << " of all its components **** ERROR ****\n";
      } else {
        edm::LogWarning("HGCalGeom") << "Thickness of the partition " << layerThick[i] << " does not match with "
                                     << thickTot << " of the components\n";
      }
    }
  }  // End of loop over blocks

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << copies.size() << " different wafer copy numbers";
  int k(0);
  for (std::unordered_set<int>::const_iterator itr = copies.begin(); itr != copies.end(); ++itr, ++k)
    edm::LogVerbatim("HGCalGeom") << "Copy[" << k << "] : " << (*itr);
  edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalModule construction ...";
#endif

  return cms::s_executed;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalModule, algorithm)
