/*
 * DDHGCalWaferF.cc
 *
 *  Created on: 09-Jan-2021
 */

#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"

#include <string>
#include <vector>
#include <sstream>

//#define EDM_ML_DEBUG

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
#ifdef EDM_ML_DEBUG
  static constexpr double f2mm = (1.0 / dd4hep::mm);
#endif
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);
  std::string motherName = args.parentName();
  const auto& material = args.value<std::string>("ModuleMaterial");
  const auto& thick = args.value<double>("ModuleThickness");
  const auto& waferSize = args.value<double>("WaferSize");
  const auto& waferSepar = args.value<double>("SensorSeparation");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferF: Module " << motherName << " made of " << material << " T "
                                << (f2mm * thick) << " Wafer 2r " << (f2mm * waferSize) << " Half Separation "
                                << (f2mm * waferSepar);
#endif
  const auto& layerNames = args.value<std::vector<std::string>>("LayerNames");
  const auto& materials = args.value<std::vector<std::string>>("LayerMaterials");
  const auto& layerThick = args.value<std::vector<double>>("LayerThickness");
  const auto& layerType = args.value<std::vector<int>>("LayerTypes");
  std::vector<int> copyNumber(materials.size(), 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferF: " << layerNames.size() << " types of volumes";
  for (unsigned int i = 0; i < layerNames.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << layerNames[i] << " of thickness "
                                  << (f2mm * layerThick[i]) << " filled with " << materials[i] << " type "
                                  << layerType[i];
#endif
  const auto& layers = args.value<std::vector<int>>("Layers");
#ifdef EDM_ML_DEBUG
  std::ostringstream st1;
  for (unsigned int i = 0; i < layers.size(); ++i)
    st1 << " [" << i << "] " << layers[i];
  edm::LogVerbatim("HGCalGeom") << "There are " << layers.size() << " blocks" << st1.str();
#endif
  const auto& nCells = args.value<int>("NCells");
  const auto& cellType = args.value<int>("CellType");
  const auto& cellNames = args.value<std::vector<std::string>>("CellNames");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferF: Cells/Wafer " << nCells << " Cell Type " << cellType << " NameSpace "
                                << ns.name() << " # of cells " << cellNames.size();
  for (unsigned int k = 0; k < cellNames.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferF: Cell[" << k << "] " << cellNames[k];
  int counter(0);
#endif

  static constexpr double tol = 0.00001;
  static const double sqrt3 = std::sqrt(3.0);
  double rM = 0.5 * (waferSize + waferSepar);
  double RM2 = rM / sqrt3;
  double R = waferSize / (3.0 * nCells);
  double r = 0.5 * R * sqrt3;
  double r2 = 0.5 * waferSize;
  double R2 = r2 / sqrt3;

  // Mother Module
  std::vector<double> xM = {rM, 0, -rM, -rM, 0, rM};
  std::vector<double> yM = {RM2, 2 * RM2, RM2, -RM2, -2 * RM2, -RM2};
  std::vector<double> zw = {-0.5 * thick, 0.5 * thick};
  std::vector<double> zx(2, 0), zy(2, 0), scale(2, 1.0);

  dd4hep::Material matter = ns.material(material);
  dd4hep::Solid solid = dd4hep::ExtrudedPolygon(xM, yM, zw, zx, zy, scale);
  ns.addSolidNS(ns.prepend(motherName), solid);
  dd4hep::Volume glogM = dd4hep::Volume(solid.name(), solid, matter);
  ns.addVolumeNS(glogM);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferF: " << solid.name() << " extruded polygon made of " << material
                                << " z|x|y|s (0) " << (f2mm * zw[0]) << ":" << (f2mm * zx[0]) << ":" << (f2mm * zy[0])
                                << ":" << scale[0] << " z|x|y|s (1) " << (f2mm * zw[1]) << ":" << (f2mm * zx[1]) << ":"
                                << (f2mm * zy[1]) << ":" << scale[1] << " and " << xM.size() << " edges";
  for (unsigned int k = 0; k < xM.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << (f2mm * xM[k]) << ":" << (f2mm * yM[k]);
#endif

  // Then the layers
  dd4hep::Rotation3D rotation;
  std::vector<double> xL = {r2, 0, -r2, -r2, 0, r2};
  std::vector<double> yL = {R2, 2 * R2, R2, -R2, -2 * R2, -R2};
  std::vector<dd4hep::Volume> glogs(materials.size());
  double zi(-0.5 * thick), thickTot(0.0);
  for (unsigned int l = 0; l < layers.size(); l++) {
    unsigned int i = layers[l];
    if (copyNumber[i] == 1) {
      zw[0] = -0.5 * layerThick[i];
      zw[1] = 0.5 * layerThick[i];
      solid = dd4hep::ExtrudedPolygon(xL, yL, zw, zx, zy, scale);
      ns.addSolidNS(ns.prepend(layerNames[i]), solid);
      matter = ns.material(materials[i]);
      glogs[i] = dd4hep::Volume(solid.name(), solid, matter);
      ns.addVolumeNS(glogs[i]);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferF: " << solid.name() << " extruded polygon made of " << materials[i]
                                    << " z|x|y|s (0) " << (f2mm * zw[0]) << ":" << (f2mm * zx[0]) << ":"
                                    << (f2mm * zy[0]) << ":" << scale[0] << " z|x|y|s (1) " << (f2mm * zw[1]) << ":"
                                    << (f2mm * zx[1]) << ":" << (f2mm * zy[1]) << ":" << scale[1] << " and "
                                    << xM.size() << " edges";
      for (unsigned int k = 0; k < xL.size(); ++k)
        edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << (f2mm * xL[k]) << ":" << (f2mm * yL[k]);
#endif
    }
    dd4hep::Position tran0(0, 0, (zi + 0.5 * layerThick[i]));
    glogM.placeVolume(glogs[i], copyNumber[i], tran0);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferF: " << glogs[i].name() << " number " << copyNumber[i]
                                  << " positioned in " << glogM.name() << " at " << tran0 << " with no rotation";
#endif
    ++copyNumber[i];
    zi += layerThick[i];
    thickTot += layerThick[i];

    if (layerType[i] > 0) {
      for (int u = 0; u < 2 * nCells; ++u) {
        for (int v = 0; v < 2 * nCells; ++v) {
          if (((v - u) < nCells) && (u - v) <= nCells) {
#ifdef EDM_ML_DEBUG
            counter++;
#endif
            int n2 = nCells / 2;
            double yp = (u - 0.5 * v - n2) * 2 * r;
            double xp = (1.5 * (v - nCells) + 1.0) * R;
            int cell(0);
            if ((u == 0) && (v == 0))
              cell = 7;
            else if ((u == 0) && (v == nCells - 1))
              cell = 8;
            else if ((u == nCells) && (v == 2 * nCells - 1))
              cell = 9;
            else if ((u == 2 * nCells - 1) && (v == 2 * nCells - 1))
              cell = 10;
            else if ((u == 2 * nCells - 1) && (v == nCells - 1))
              cell = 11;
            else if ((u == nCells) && (v == 0))
              cell = 12;
            else if (u == 0)
              cell = 1;
            else if ((v - u) == (nCells - 1))
              cell = 4;
            else if (v == (2 * nCells - 1))
              cell = 2;
            else if (u == (2 * nCells - 1))
              cell = 5;
            else if ((u - v) == nCells)
              cell = 3;
            else if (v == 0)
              cell = 6;
            dd4hep::Position tran(xp, yp, 0);
            int copy = HGCalTypes::packCellTypeUV(cellType, u, v);
            glogs[i].placeVolume(ns.volume(cellNames[cell]), copy, dd4hep::Transform3D(rotation, tran));
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom")
                << "DDHGCalWaferF: " << cellNames[cell] << " number " << copy << " positioned in " << glogs[i].name()
                << " at (" << (f2mm * xp) << ", " << (f2mm * yp) << ",0)  with " << rotation;
#endif
          }
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "\nDDHGCalWaferF::Counter : " << counter << "\n===============================\n";
#endif
  if (std::abs(thickTot - thick) >= tol) {
    if (thickTot > thick) {
      edm::LogError("HGCalGeom") << "Thickness of the partition " << thick << " is smaller than " << thickTot
                                 << ": thickness of all its components **** ERROR ****";
    } else {
      edm::LogWarning("HGCalGeom") << "Thickness of the partition " << thick << " does not match with " << thickTot
                                   << " of the components";
    }
  }

  return cms::s_executed;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalWaferF, algorithm)
