/*
 * DDAHcalModuleAlgo.cc
 *
 *  Created on: 27-August-2019
 *      Author: Sunanda Banerjee
 */

#include "DataFormats/Math/interface/CMSUnits.h"
#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HGCalCommonData/interface/AHCalParameters.h"

//#define EDM_ML_DEBUG
using namespace cms_units::operators;

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);
  static constexpr double tol = 0.00001;

  const auto& tile = args.value<std::string>("TileName");                          // Scintillator tile
  const auto& materials = args.value<std::vector<std::string> >("MaterialNames");  // Materials
  const auto& names = args.value<std::vector<std::string> >("VolumeNames");        // Names
  const auto& thick = args.value<std::vector<double> >("Thickness");               // Thickness of the material
  std::vector<int> copyNumber;                                                     // Initial copy numbers
  copyNumber.resize(materials.size(), 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: Tile " << tile;
  edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << materials.size() << " types of volumes";
  for (unsigned int i = 0; i < names.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << names[i] << " of thickness " << thick[i]
                                  << " filled with " << materials[i] << " first copy number " << copyNumber[i];
#endif
  const auto& layers = args.value<std::vector<int> >("Layers");             // Number of layers in a section
  const auto& layerThick = args.value<std::vector<double> >("LayerThick");  // Thickness of each section
  const auto& layerType = args.value<std::vector<int> >("LayerType");       // Type of the layer
  const auto& layerSense = args.value<std::vector<int> >("LayerSense");     // Content of a layer (sensitive?)
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << layers.size() << " blocks";
  for (unsigned int i = 0; i < layers.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << layerThick[i] << " with " << layers[i]
                                  << " layers";
  edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << layerType.size() << " layers";
  for (unsigned int i = 0; i < layerType.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerType[i] << " sensitive class "
                                  << layerSense[i];
#endif
  const auto& widths = args.value<std::vector<double> >("Widths");      // Width (passive, active)
  const auto& heights = args.value<std::vector<double> >("Heights");    // Heights (passive, active)
  const auto& tileN = args.value<std::vector<int> >("TileN");           // # of tiles (along x, y)
  const auto& tileStep = args.value<std::vector<double> >("TileStep");  // Separation between tiles (x, y)
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << widths.size() << " sizes for width "
                                << "and height:";
  for (unsigned int i = 0; i < widths.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << " [" << i << "] " << widths[i] << ":" << heights[i];
  edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << tileN.size() << " tile positioning parameters";
  for (unsigned int i = 0; i < tileN.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << " [" << i << "] " << tileN[i] << ":" << tileStep[i];
#endif
  const auto& zMinBlock = args.value<double>("zMinBlock");  // Starting z-value of the block
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: zStart " << zMinBlock << "  NameSpace " << ns.name();
#endif

  // Mother module
  dd4hep::Volume module = ns.volume(args.parentName());

  double zi(zMinBlock);
  int laymin(0);
  for (unsigned int i = 0; i < layers.size(); i++) {
    double zo = zi + layerThick[i];
    int laymax = laymin + layers[i];
    double zz = zi;
    double thickTot(0);
    for (int ly = laymin; ly < laymax; ++ly) {
      int ii = layerType[ly];
      int copy = copyNumber[ii];
      zz += (0.5 * thick[ii]);
      thickTot += thick[ii];

      std::string name = "HGCal" + names[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo test: Layer " << ly << ":" << ii << " Front " << zi
                                    << " Back " << zo << " superlayer thickness " << layerThick[i];
#endif
      dd4hep::Material matter = ns.material(materials[ii]);
      dd4hep::Volume glog;
      if (layerSense[ly] == 0) {
        dd4hep::Solid solid = dd4hep::Box(0.5 * widths[0], 0.5 * heights[0], 0.5 * thick[ii]);
        ns.addSolidNS(ns.prepend(name), solid);
        glog = dd4hep::Volume(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << solid.name() << " Box made of " << materials[ii]
                                      << " of dimensions " << 0.5 * widths[0] << ", " << 0.5 * heights[0] << ", "
                                      << 0.5 * thick[ii];
#endif
      } else {
        dd4hep::Solid solid = dd4hep::Box(0.5 * widths[1], 0.5 * heights[1], 0.5 * thick[ii]);
        ns.addSolidNS(ns.prepend(name), solid);
        glog = dd4hep::Volume(solid.name(), solid, matter);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << solid.name() << " Box made of " << materials[ii]
                                      << " of dimensions " << 0.5 * widths[1] << ", " << 0.5 * heights[1] << ", "
                                      << 0.5 * thick[ii];
#endif
        int ncol = tileN[0] / 2;
        int nrow = tileN[1] / 2;
#ifdef EDM_ML_DEBUG
        int kount(0);
        edm::LogVerbatim("HGCalGeom") << glog.name() << " Row " << nrow << " Column " << ncol;
#endif
        for (int nr = -nrow; nr <= nrow; ++nr) {
          int inr = (nr >= 0) ? nr : -nr;
          double ypos = (nr >= 0) ? (inr - 0.5) * tileStep[1] : -(inr - 0.5) * tileStep[1];
          for (int nc = -ncol; nc <= ncol; ++nc) {
            int inc = (nc >= 0) ? nc : -nc;
            double xpos = (nc >= 0) ? (inc - 0.5) * tileStep[0] : -(inc - 0.5) * tileStep[0];
            if (nr != 0 && nc != 0) {
              dd4hep::Position tran(xpos, ypos, 0.0);
              dd4hep::Rotation3D rotation;
              int copy = inr * AHCalParameters::kColumn_ + inc;
              if (nc < 0)
                copy += AHCalParameters::kRowColumn_;
              if (nr < 0)
                copy += AHCalParameters::kSignRowColumn_;
              dd4hep::Volume glog1 = ns.volume(tile);
              glog.placeVolume(glog1, copy, dd4hep::Transform3D(rotation, tran));
#ifdef EDM_ML_DEBUG
              kount++;
              edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << tile << " number " << copy << " positioned in "
                                            << glog.name() << " at " << tran << " with " << rotation;
#endif
            }
          }
        }
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << kount << " tiles for " << glog.name();
#endif
      }
      dd4hep::Position r1(0, 0, zz);
      module.placeVolume(glog, copy, r1);
      ++copyNumber[ii];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << glog.name() << " number " << copy << " positioned in "
                                    << module.name() << " at " << r1 << " with no rotation";
#endif
      zz += (0.5 * thick[ii]);
    }  // End of loop over layers in a block
    zi = zo;
    laymin = laymax;
    if (fabs(thickTot - layerThick[i]) > tol) {
      if (thickTot > layerThick[i]) {
        edm::LogError("HGCalGeom") << "Thickness of the partition " << layerThick[i] << " is smaller than thickness "
                                   << thickTot << " of all its components **** ERROR ****\n";
      } else {
        edm::LogWarning("HGCalGeom") << "Thickness of the partition " << layerThick[i] << " does not match with "
                                     << thickTot << " of the components\n";
      }
    }
  }  // End of loop over blocks

  return cms::s_executed;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDAHcalModuleAlgo, algorithm)
