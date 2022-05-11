/*
 * DDHGCalWaferPartialRotated.cc
 *
 *  Created on: 09-Jan-2021
 */

#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DetectorDescription/DDCMS/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/HGCalCommonData/interface/HGCalCell.h"

#include <string>
#include <vector>
#include <sstream>

//#define EDM_ML_DEBUG

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);
  std::string parentName = args.parentName();
  const auto& material = args.value<std::string>("ModuleMaterial");
  const auto& thick = args.value<double>("ModuleThickness");
  const auto& waferSize = args.value<double>("WaferSize");
  const auto& waferThick = args.value<double>("WaferThickness");
  const auto& waferTag = args.value<std::string>("WaferTag");
#ifdef EDM_ML_DEBUG
  const auto& waferSepar = args.value<double>("SensorSeparation");
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferPartialRotated: Module " << parentName << " made of " << material
                                << " T " << cms::convert2mm(thick) << " Wafer 2r " << cms::convert2mm(waferSize)
                                << " Half Separation " << cms::convert2mm(waferSepar) << " T "
                                << cms::convert2mm(waferThick);
#endif
  const auto& tags = args.value<std::vector<std::string>>("Tags");
  const auto& partialTypes = args.value<std::vector<int>>("PartialTypes");
  const auto& placementIndex = args.value<std::vector<int>>("PlacementIndex");
  const auto& placementIndexTags = args.value<std::vector<std::string>>("PlacementIndexTags");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferPartialRotated: " << tags.size() << " variations of wafer types";
  for (unsigned int k = 0; k < tags.size(); ++k) {
    for (unsigned int m = 0; m < placementIndex.size(); ++m) {
      edm::LogVerbatim("HGCalGeom") << "Type[" << k << "] " << tags[k] << " Partial " << partialTypes[k]
                                    << " Placement Index " << placementIndex[m] << " Tag " << placementIndexTags[m];
    }
  }
#endif
  const auto& layerNames = args.value<std::vector<std::string>>("LayerNames");
  const auto& materials = args.value<std::vector<std::string>>("LayerMaterials");
  const auto& layerThick = args.value<std::vector<double>>("LayerThickness");
  const auto& layerType = args.value<std::vector<int>>("LayerTypes");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferPartialRotated: " << layerNames.size() << " types of volumes";
  for (unsigned int i = 0; i < layerNames.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << layerNames[i] << " of thickness "
                                  << cms::convert2mm(layerThick[i]) << " filled with " << materials[i] << " type "
                                  << layerType[i];
#endif
  const auto& layers = args.value<std::vector<int>>("Layers");
#ifdef EDM_ML_DEBUG
  std::ostringstream st1;
  for (unsigned int i = 0; i < layers.size(); ++i)
    st1 << " [" << i << "] " << layers[i];
  edm::LogVerbatim("HGCalGeom") << "There are " << layers.size() << " blocks" << st1.str();
#endif
  const auto& senseName = args.value<std::string>("SenseName");
  const auto& senseT = args.value<double>("SenseThick");
  const auto& senseType = args.value<int>("SenseType");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferPartialRotated: NameSpace " << ns.name() << " Sensitive Layer Name "
                                << senseName << " Thickness " << senseT << " Type " << senseType;
#endif

  static constexpr double tol = 0.00001 * dd4hep::mm;
  static const double sqrt3 = std::sqrt(3.0);
  double r = 0.5 * waferSize;
  double R = 2.0 * r / sqrt3;

  // Loop over all types
  for (unsigned int k = 0; k < tags.size(); ++k) {
    for (unsigned int m = 0; m < placementIndex.size(); ++m) {
      // First the mother
      std::string mother = parentName + placementIndexTags[m] + waferTag + tags[k];
      std::vector<std::pair<double, double>> wxy =
          HGCalWaferMask::waferXY(partialTypes[k], placementIndex[m], r, R, 0.0, 0.0);
      std::vector<double> xM, yM;
      for (unsigned int i = 0; i < (wxy.size() - 1); ++i) {
        xM.emplace_back(wxy[i].first);
        yM.emplace_back(wxy[i].second);
      }
      std::vector<double> zw = {-0.5 * thick, 0.5 * thick};
      std::vector<double> zx(2, 0), zy(2, 0), scale(2, 1.0);

      dd4hep::Material matter = ns.material(material);
      dd4hep::Solid solid = dd4hep::ExtrudedPolygon(xM, yM, zw, zx, zy, scale);
      ns.addSolidNS(ns.prepend(mother), solid);
      dd4hep::Volume glogM = dd4hep::Volume(solid.name(), solid, matter);
      ns.addVolumeNS(glogM);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferPartialRotated: " << solid.name() << " extruded polygon made of "
                                    << material << " z|x|y|s (0) " << cms::convert2mm(zw[0]) << ":"
                                    << cms::convert2mm(zx[0]) << ":" << cms::convert2mm(zy[0]) << ":" << scale[0]
                                    << " z|x|y|s (1) " << cms::convert2mm(zw[1]) << ":" << cms::convert2mm(zx[1]) << ":"
                                    << cms::convert2mm(zy[1]) << ":" << scale[1] << " partial " << partialTypes[k]
                                    << " placement index " << placementIndex[m] << " and " << xM.size() << " edges";
      for (unsigned int j = 0; j < xM.size(); ++j)
        edm::LogVerbatim("HGCalGeom") << "[" << j << "] " << cms::convert2mm(xM[j]) << ":" << cms::convert2mm(yM[j]);
#endif

      // Then the layers
      dd4hep::Rotation3D rotation;
      wxy = HGCalWaferMask::waferXY(partialTypes[k], placementIndex[m], r, R, 0.0, 0.0);
      std::vector<double> xL, yL;
      for (unsigned int i = 0; i < (wxy.size() - 1); ++i) {
        xL.emplace_back(wxy[i].first);
        yL.emplace_back(wxy[i].second);
      }
      std::vector<dd4hep::Volume> glogs(materials.size());
      std::vector<int> copyNumber(materials.size(), 1);
      double zi(-0.5 * thick), thickTot(0.0);
      for (unsigned int l = 0; l < layers.size(); l++) {
        unsigned int i = layers[l];
        if (copyNumber[i] == 1) {
          if (layerType[i] > 0) {
            zw[0] = -0.5 * waferThick;
            zw[1] = 0.5 * waferThick;
          } else {
            zw[0] = -0.5 * layerThick[i];
            zw[1] = 0.5 * layerThick[i];
          }
          solid = dd4hep::ExtrudedPolygon(xL, yL, zw, zx, zy, scale);
          std::string lname = layerNames[i] + placementIndexTags[m] + waferTag + tags[k];
          ns.addSolidNS(ns.prepend(lname), solid);
          matter = ns.material(materials[i]);
          glogs[i] = dd4hep::Volume(solid.name(), solid, matter);
          ns.addVolumeNS(glogs[i]);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferPartialRotated: " << solid.name()
                                        << " extruded polygon made of " << materials[i] << " z|x|y|s (0) "
                                        << cms::convert2mm(zw[0]) << ":" << cms::convert2mm(zx[0]) << ":"
                                        << cms::convert2mm(zy[0]) << ":" << scale[0] << " z|x|y|s (1) "
                                        << cms::convert2mm(zw[1]) << ": partial " << partialTypes[k]
                                        << " placement index " << placementIndex[m] << cms::convert2mm(zx[1]) << ":"
                                        << cms::convert2mm(zy[1]) << ":" << scale[1] << " and " << xM.size()
                                        << " edges";
          for (unsigned int j = 0; j < xL.size(); ++j)
            edm::LogVerbatim("HGCalGeom")
                << "[" << j << "] " << cms::convert2mm(xL[j]) << ":" << cms::convert2mm(yL[j]);
#endif
        }
        if (layerType[i] > 0) {
          std::string sname = senseName + placementIndexTags[m] + waferTag + tags[k];
          zw[0] = -0.5 * senseT;
          zw[1] = 0.5 * senseT;
          solid = dd4hep::ExtrudedPolygon(xL, yL, zw, zx, zy, scale);
          ns.addSolidNS(ns.prepend(sname), solid);
          dd4hep::Volume glog = dd4hep::Volume(solid.name(), solid, matter);
          ns.addVolumeNS(glog);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferPartialRotated: " << solid.name()
                                        << " extruded polygon made of " << materials[i] << " z|x|y|s (0) "
                                        << cms::convert2mm(zw[0]) << ":" << cms::convert2mm(zx[0]) << ":"
                                        << cms::convert2mm(zy[0]) << ":" << scale[0] << " z|x|y|s (1) "
                                        << cms::convert2mm(zw[1]) << ":" << cms::convert2mm(zx[1]) << ":"
                                        << cms::convert2mm(zy[1]) << ":" << scale[1] << " partial " << partialTypes[k]
                                        << " placement index " << placementIndex[m] << " and " << xL.size() << " edges";
          for (unsigned int j = 0; j < xL.size(); ++j)
            edm::LogVerbatim("HGCalGeom")
                << "[" << j << "] " << cms::convert2mm(xL[j]) << ":" << cms::convert2mm(yL[j]);
#endif
          auto posSense = HGCalCell::cellOrient(placementIndex[m]);
          double zpos = (posSense.second < 0) ? -0.5 * (waferThick - senseT) : 0.5 * (waferThick - senseT);
          dd4hep::Position tran(0, 0, zpos);
          int copy = 10 + senseType;
          glogs[i].placeVolume(glog, copy, tran);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferPartialRotated: " << glog.name() << " number " << copy
                                        << " positioned in " << glogs[i].name() << " at (0, 0," << cms::convert2mm(zpos)
                                        << ") with no rotation";
#endif
        }
        dd4hep::Position tran0(0, 0, (zi + 0.5 * layerThick[i]));
        glogM.placeVolume(glogs[i], copyNumber[i], tran0);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferPartialRotated: " << glogs[i].name() << " number "
                                      << copyNumber[i] << " positioned in " << glogM.name() << " at (0,0,"
                                      << cms::convert2mm(zi + 0.5 * layerThick[i]) << " with no rotation";
#endif
        ++copyNumber[i];
        zi += layerThick[i];
        thickTot += layerThick[i];
      }
      if (std::abs(thickTot - thick) >= tol) {
        if (thickTot > thick) {
          edm::LogError("HGCalGeom") << "Thickness of the partition " << cms::convert2mm(thick) << " is smaller than "
                                     << cms::convert2mm(thickTot)
                                     << ": thickness of all its components **** ERROR ****";
        } else {
          edm::LogWarning("HGCalGeom") << "Thickness of the partition " << cms::convert2mm(thick)
                                       << " does not match with " << cms::convert2mm(thickTot) << " of the components";
        }
      }
    }
  }

  return cms::s_executed;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalWaferPartialRotated, algorithm)
