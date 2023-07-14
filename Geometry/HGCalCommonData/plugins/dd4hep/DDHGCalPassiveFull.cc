///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalPassiveFull.cc
// Description: Geometry factory class for the passive part of a full silicon
//              module
// Created by Sunanda Banerjee
///////////////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>
#include <sstream>

#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/angle_units.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DetectorDescription/DDCMS/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG
using namespace angle_units::operators;

struct HGCalPassiveFull {
  HGCalPassiveFull() { throw cms::Exception("HGCalGeom") << "Wrong initialization to HGCalPassiveFull"; }
  HGCalPassiveFull(cms::DDParsingContext& ctxt, xml_h e) {
    cms::DDNamespace ns(ctxt, e, true);
    cms::DDAlgoArguments args(ctxt, e);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalPassiveFull: Creating an instance";
#endif
    std::string parentName = args.parentName();
    std::string material = args.value<std::string>("ModuleMaterial");
    double thick = args.value<double>("ModuleThickness");
    double waferSize = args.value<double>("WaferSize");
    double waferSepar = args.value<double>("SensorSeparation");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalPassiveFull: Module " << parentName << " made of " << material << " T "
                                  << cms::convert2mm(thick) << " Wafer 2r " << cms::convert2mm(waferSize) << " Half Separation " << cms::convert2mm(waferSepar);
#endif
    std::vector<std::string> layerNames = args.value<std::vector<std::string>>("LayerNames");
    std::vector<std::string> materials = args.value<std::vector<std::string>>("LayerMaterials");
    std::vector<double> layerThick = args.value<std::vector<double>>("LayerThickness");
    std::vector<int> copyNumber;
    copyNumber.resize(materials.size(), 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalPassiveFull: " << layerNames.size() << " types of volumes";
    for (unsigned int i = 0; i < layerNames.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << layerNames[i] << " of thickness " << cms::convert2mm(layerThick[i])
                                    << " filled with " << materials[i];
#endif
    std::vector<int> layerType = args.value<std::vector<int>>("LayerType");
#ifdef EDM_ML_DEBUG
    std::ostringstream st1;
    for (unsigned int i = 0; i < layerType.size(); ++i)
      st1 << " [" << i << "] " << layerType[i];
    edm::LogVerbatim("HGCalGeom") << "There are " << layerType.size() << " blocks" << st1.str();

    edm::LogVerbatim("HGCalGeom") << "==>> Executing DDHGCalPassiveFull...";
#endif

    static constexpr double tol = 0.00001;
    static const double sqrt3 = std::sqrt(3.0);
    double rM = 0.5 * (waferSize + waferSepar);
    double RM2 = rM / sqrt3;

    // First the mother
    std::vector<double> xM = {rM, 0, -rM, -rM, 0, rM};
    std::vector<double> yM = {RM2, 2 * RM2, RM2, -RM2, -2 * RM2, -RM2};
    std::vector<double> zw = {-0.5 * thick, 0.5 * thick};
    std::vector<double> zx(2, 0), zy(2, 0), scale(2, 1.0);
    dd4hep::Solid solid = dd4hep::ExtrudedPolygon(xM, yM, zw, zx, zy, scale);
    ns.addSolidNS(ns.prepend(parentName), solid);
    dd4hep::Material matter = ns.material(material);
    dd4hep::Volume glogM = dd4hep::Volume(solid.name(), solid, matter);
    ns.addVolumeNS(glogM);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalPassiveFull: " << solid.name() << " extruded polygon made of "
                                  << matter.name() << " z|x|y|s (0) " << cms::convert2mm(zw[0]) << ":" << cms::convert2mm(zx[0]) << ":" << cms::convert2mm(zy[0]) << ":"
                                  << scale[0] << " z|x|y|s (1) " << cms::convert2mm(zw[1]) << ":" << cms::convert2mm(zx[1]) << ":" << cms::convert2mm(zy[1]) << ":"
                                  << scale[1] << " and " << xM.size() << " edges";
    for (unsigned int kk = 0; kk < xM.size(); ++kk)
      edm::LogVerbatim("HGCalGeom") << "[" << kk << "] " << cms::convert2mm(xM[kk]) << ":" << cms::convert2mm(yM[kk]);
#endif

    // Then the layers
    std::vector<dd4hep::Volume> glogs(materials.size());
    double zi(-0.5 * thick), thickTot(0.0);
    for (unsigned int l = 0; l < layerType.size(); l++) {
      unsigned int i = layerType[l];
      if (copyNumber[i] == 1) {
        zw[0] = -0.5 * layerThick[i];
        zw[1] = 0.5 * layerThick[i];
        std::string layerName = parentName + layerNames[i];
        solid = dd4hep::ExtrudedPolygon(xM, yM, zw, zx, zy, scale);
        ns.addSolidNS(ns.prepend(layerName), solid);
        matter = ns.material(materials[i]);
        glogs[i] = dd4hep::Volume(solid.name(), solid, matter);
        ns.addVolumeNS(glogs[i]);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalPassiveFull: Layer " << i << ":" << l << ":" << solid.name()
                                      << " extruded polygon made of " << matter.name() << " z|x|y|s (0) " << cms::convert2mm(zw[0])
                                      << ":" << cms::convert2mm(zx[0]) << ":" << cms::convert2mm(zy[0]) << ":" << scale[0] << " z|x|y|s (1) " << cms::convert2mm(zw[1])
                                      << ":" << cms::convert2mm(zx[1]) << ":" << cms::convert2mm(zy[1]) << ":" << scale[1] << " and " << xM.size()
                                      << " edges";
        for (unsigned int kk = 0; kk < xM.size(); ++kk)
          edm::LogVerbatim("HGCalGeom") << "[" << kk << "] " << cms::convert2mm(xM[kk]) << ":" << cms::convert2mm(yM[kk]);
#endif
      }
      dd4hep::Position tran0(0, 0, (zi + 0.5 * layerThick[i]));
      glogM.placeVolume(glogs[i], copyNumber[i], tran0);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalPassiveFull: " << glogs[i].name() << " number " << copyNumber[i]
                                    << " positioned in " << glogM.name() << " at (0, 0, " << cms::convert2mm(zi + 0.5 * layerThick[i]) << ") with no rotation";
#endif
      ++copyNumber[i];
      zi += layerThick[i];
      thickTot += layerThick[i];
    }
    if ((std::abs(thickTot - thick) >= tol) && (!layerType.empty())) {
      if (thickTot > thick) {
        edm::LogError("HGCalGeom") << "Thickness of the partition " << thick << " is smaller than " << thickTot
                                   << ": thickness of all its components **** ERROR ****";
      } else {
        edm::LogWarning("HGCalGeom") << "Thickness of the partition " << thick << " does not match with " << thickTot
                                     << " of the components";
      }
    }
  }
};

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  HGCalPassiveFull passiveFullAlgo(ctxt, e);
  return cms::s_executed;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalPassiveFull, algorithm)
