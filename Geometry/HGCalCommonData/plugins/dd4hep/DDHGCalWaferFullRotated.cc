///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalWaferFullRotated.cc
// Description: Geometry factory class for a full silicon Wafer
// Created by Sunanda Banerjee, Pruthvi Suryadevara, Indranil Das
///////////////////////////////////////////////////////////////////////////////
#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DetectorDescription/DDCMS/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalCell.h"

#include <string>
#include <vector>
#include <sstream>

//#define EDM_ML_DEBUG

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: Creating an instance";
#endif
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);

  const auto& material = args.value<std::string>("ModuleMaterial");
  const auto& thick = args.value<double>("ModuleThickness");
  const auto& waferSize = args.value<double>("WaferSize");
  const auto& waferSepar = args.value<double>("SensorSeparation");
  const auto& waferThick = args.value<double>("WaferThickness");
  const auto& waferTag = args.value<std::string>("WaferTag");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: Module " << args.parentName() << " made of " << material
                                << " T " << thick << " Wafer 2r " << waferSize << " Half Separation " << waferSepar
                                << " T " << waferThick;
#endif
  const auto& orient = args.value<std::vector<int> >("WaferOrient");
  const auto& face = args.value<std::vector<int> >("WaferFace");
  const auto& tag = args.value<std::vector<std::string> >("WaferPlacementIndex");
  const auto& layerNames = args.value<std::vector<std::string> >("LayerNames");
  const auto& materials = args.value<std::vector<std::string> >("LayerMaterials");
  const auto& layerThick = args.value<std::vector<double> >("LayerThickness");
  const auto& layerType = args.value<std::vector<int> >("LayerTypes");
  std::vector<int> copyNumber;
  copyNumber.resize(materials.size(), 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: " << layerNames.size() << " types of volumes";
  for (unsigned int i = 0; i < layerNames.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << layerNames[i] << " of thickness " << layerThick[i]
                                  << " filled with " << materials[i] << " type " << layerType[i];
#endif
  const auto& layers = args.value<std::vector<int> >("Layers");
#ifdef EDM_ML_DEBUG
  std::ostringstream st1;
  for (unsigned int i = 0; i < layers.size(); ++i)
    st1 << " [" << i << "] " << layers[i];
  edm::LogVerbatim("HGCalGeom") << "There are " << layers.size() << " blocks" << st1.str();
#endif
  const auto& nCells = args.value<int>("NCells");
  int cellType(-1);
  std::vector<std::string> cellNames;
  std::vector<int> cellOffset;
  if (nCells > 0) {
    cellType = args.value<int>("CellType");
    cellNames = args.value<std::vector<std::string> >("CellNames");
    cellOffset = args.value<std::vector<int> >("CellOffset");
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: Cells/Wafer " << nCells << " Cell Type " << cellType
                                << " # of cells " << cellNames.size();
  std::ostringstream st2;
  for (unsigned int i = 0; i < cellOffset.size(); ++i)
    st2 << " [" << i << "] " << cellOffset[i];
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: " << cellOffset.size() << " types of cells with offsets "
                                << st2.str();
  for (unsigned int k = 0; k < cellNames.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: Cell[" << k << "] " << cellNames[k];

  edm::LogVerbatim("HGCalGeom") << "==>> Executing DDHGCalWaferFullRotated...";
#endif

  static constexpr double tol = 0.00001;
  static const double sqrt3 = std::sqrt(3.0);
  double rM = 0.5 * waferSize;
  double RM2 = rM / sqrt3;
  double r2 = 0.5 * waferSize;
  double R2 = r2 / sqrt3;
  const int nFine(nCells), nCoarse(nCells);
  HGCalCell wafer((waferSize + waferSepar), nFine, nCoarse);
  for (unsigned int k = 0; k < tag.size(); ++k) {
    // First the mother
    std::vector<double> xM = {rM, 0, -rM, -rM, 0, rM};
    std::vector<double> yM = {RM2, 2 * RM2, RM2, -RM2, -2 * RM2, -RM2};
    std::vector<double> zw = {-0.5 * thick, 0.5 * thick};
    std::vector<double> zx(2, 0), zy(2, 0), scale(2, 1.0);
    std::string parentName = args.parentName() + tag[k] + waferTag;
    dd4hep::Material matter = ns.material(material);
    dd4hep::Solid solid = dd4hep::ExtrudedPolygon(xM, yM, zw, zx, zy, scale);
    ns.addSolidNS(ns.prepend(parentName), solid);
    dd4hep::Volume glogM = dd4hep::Volume(solid.name(), solid, matter);
    ns.addVolumeNS(glogM);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: " << solid.name() << " extruded polygon made of "
                                  << material << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":"
                                  << scale[0] << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":"
                                  << scale[1] << " and " << xM.size() << " edges";
    for (unsigned int kk = 0; kk < xM.size(); ++kk)
      edm::LogVerbatim("HGCalGeom") << "[" << kk << "] " << xM[kk] << ":" << yM[kk];
#endif

    // Then the layers
    std::vector<double> xL = {r2, 0, -r2, -r2, 0, r2};
    std::vector<double> yL = {R2, 2 * R2, R2, -R2, -2 * R2, -R2};
    std::vector<dd4hep::Volume> glogs(materials.size());
    for (unsigned int ii = 0; ii < copyNumber.size(); ii++) {
      copyNumber[ii] = 1;
    }
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
        std::string layerName = layerNames[i] + tag[k] + waferTag;
        solid = dd4hep::ExtrudedPolygon(xL, yL, zw, zx, zy, scale);
        ns.addSolidNS(ns.prepend(layerName), solid);
        matter = ns.material(materials[i]);
        glogs[i] = dd4hep::Volume(solid.name(), solid, matter);
        ns.addVolumeNS(glogs[i]);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: " << solid.name() << " extruded polygon made of "
                                      << materials[i] << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":"
                                      << scale[0] << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":"
                                      << scale[1] << " and " << xL.size() << " edges";
        for (unsigned int kk = 0; kk < xL.size(); ++kk)
          edm::LogVerbatim("HGCalGeom") << "[" << kk << "] " << xL[kk] << ":" << yL[kk];
#endif
      }
      dd4hep::Position tran0(0, 0, (zi + 0.5 * layerThick[i]));
      glogM.placeVolume(glogs[i], copyNumber[i], tran0);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: " << glogs[i].name() << " number " << copyNumber[i]
                                    << " positioned in " << glogM.name() << " at " << tran0 << " with no rotation";
#endif
      ++copyNumber[i];
      zi += layerThick[i];
      thickTot += layerThick[i];
      if ((layerType[i] > 0) && (nCells > 0)) {
        for (int u = 0; u < 2 * nCells; ++u) {
          for (int v = 0; v < 2 * nCells; ++v) {
            if (((v - u) < nCells) && ((u - v) <= nCells)) {
              int placeIndex = wafer.cellPlacementIndex(1, HGCalTypes::waferFrontBack(face[k]), orient[k]);
              std::pair<double, double> xy1 = wafer.cellUV2XY1(u, v, placeIndex, cellType);
              double yp = xy1.second;
              double xp = xy1.first;
              int cell(0);
              std::pair<int, int> cell1 = wafer.cellUV2Cell(u, v, placeIndex, cellType);
              cell = cell1.first + cellOffset[cell1.second];
              dd4hep::Position tran(xp, yp, 0);
              int copy = HGCalTypes::packCellTypeUV(cellType, u, v);
              glogs[i].placeVolume(ns.volume(cellNames[cell]), copy, tran);
#ifdef EDM_ML_DEBUG
              edm::LogVerbatim("HGCalGeom")
                  << "DDHGCalWaferFullRotated: " << cellNames[cell] << " number " << copy << " positioned in "
                  << glogs[i].name() << " at " << tran << " with no rotation";
#endif
            }
          }
        }
      }
    }
    if (std::abs(thickTot - thick) >= tol) {
      if (thickTot > thick) {
        edm::LogError("HGCalGeom") << "Thickness of the partition " << thick << " is smaller than " << thickTot
                                   << ": thickness of all its components **** ERROR ****";
      } else {
        edm::LogWarning("HGCalGeom") << "Thickness of the partition " << thick << " does not match with " << thickTot
                                     << " of the components";
      }
    }
  }
  return cms::s_executed;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalWaferFullRotated, algorithm);
