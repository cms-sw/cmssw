///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalWaferFullRotated.cc
// Description: Geometry factory class for a full silicon Wafer
// Created by Sunanda Banerjee, Pruthvi Suryadevara, Indranil Das
///////////////////////////////////////////////////////////////////////////////
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalCell.h"

#include <string>
#include <vector>
#include <sstream>

//#define EDM_ML_DEBUG

class DDHGCalWaferFullRotated : public DDAlgorithm {
public:
  // Constructor and Destructor
  DDHGCalWaferFullRotated();
  ~DDHGCalWaferFullRotated() override = default;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

private:
  std::string material_;                 // Material name for module with gap
  std::string waferTag_;                 // Tag for type pf wafer
  double thick_;                         // Module thickness
  double waferSize_;                     // Wafer size
  double waferSepar_;                    // Sensor separation
  double waferThick_;                    // Wafer thickness
  std::vector<std::string> layerNames_;  // Names of the layers
  std::vector<std::string> materials_;   // Materials of the layers
  std::vector<std::string> tag_;         // Tag of placement index
  std::vector<double> layerThick_;       // Thickness of layers
  std::vector<int> layerType_;           // Layer types
  std::vector<int> copyNumber_;          // Initial copy numbers
  std::vector<int> layers_;              // Number of layers in a section
  std::vector<int> orient_;              // Orientation of wafer
  std::vector<int> face_;                // Front or back of cooling layer
  int nCells_;                           // Half number of cells along u-v axis
  int cellType_;                         // Cell Type (0,1,2: Fine, Course 2/3)
  std::vector<int> cellOffset_;          // Offset of cells of each type
  std::vector<std::string> cellNames_;   // Name of the cells
  std::string nameSpace_;                // Namespace to be used
};

DDHGCalWaferFullRotated::DDHGCalWaferFullRotated() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: Creating an instance";
#endif
}

void DDHGCalWaferFullRotated::initialize(const DDNumericArguments& nArgs,
                                         const DDVectorArguments& vArgs,
                                         const DDMapArguments&,
                                         const DDStringArguments& sArgs,
                                         const DDStringVectorArguments& vsArgs) {
  material_ = sArgs["ModuleMaterial"];
  thick_ = nArgs["ModuleThickness"];
  waferSize_ = nArgs["WaferSize"];
  waferSepar_ = nArgs["SensorSeparation"];
  waferThick_ = nArgs["WaferThickness"];
  waferTag_ = sArgs["WaferTag"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: Module " << parent().name() << " made of " << material_
                                << " T " << thick_ << " Wafer 2r " << waferSize_ << " Half Separation " << waferSepar_
                                << " T " << waferThick_;
#endif
  orient_ = dbl_to_int(vArgs["WaferOrient"]);
  face_ = dbl_to_int(vArgs["WaferFace"]);
  tag_ = vsArgs["WaferPlacementIndex"];
  layerNames_ = vsArgs["LayerNames"];
  materials_ = vsArgs["LayerMaterials"];
  layerThick_ = vArgs["LayerThickness"];
  layerType_ = dbl_to_int(vArgs["LayerTypes"]);
  copyNumber_.resize(materials_.size(), 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: " << layerNames_.size() << " types of volumes";
  for (unsigned int i = 0; i < layerNames_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << layerNames_[i] << " of thickness " << layerThick_[i]
                                  << " filled with " << materials_[i] << " type " << layerType_[i];
#endif
  layers_ = dbl_to_int(vArgs["Layers"]);
#ifdef EDM_ML_DEBUG
  std::ostringstream st1;
  for (unsigned int i = 0; i < layers_.size(); ++i)
    st1 << " [" << i << "] " << layers_[i];
  edm::LogVerbatim("HGCalGeom") << "There are " << layers_.size() << " blocks" << st1.str();
#endif
  nCells_ = (int)(nArgs["NCells"]);
  cellType_ = (int)(nArgs["CellType"]);
  cellNames_ = vsArgs["CellNames"];
  cellOffset_ = dbl_to_int(vArgs["CellOffset"]);
  nameSpace_ = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: Cells/Wafer " << nCells_ << " Cell Type " << cellType_
                                << " NameSpace " << nameSpace_ << " # of cells " << cellNames_.size();
  std::ostringstream st2;
  for (unsigned int i = 0; i < cellOffset_.size(); ++i)
    st2 << " [" << i << "] " << cellOffset_[i];
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: " << cellOffset_.size() << " types of cells with offsets "
                                << st2.str();
  for (unsigned int k = 0; k < cellNames_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: Cell[" << k << "] " << cellNames_[k];
#endif
}

void DDHGCalWaferFullRotated::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Executing DDHGCalWaferFullRotated...";
#endif

  static constexpr double tol = 0.00001;
  static const double sqrt3 = std::sqrt(3.0);
  double rM = 0.5 * waferSize_;
  double RM2 = rM / sqrt3;
  double r2 = 0.5 * waferSize_;
  double R2 = r2 / sqrt3;
  const int nFine(nCells_), nCoarse(nCells_);
  HGCalCell wafer((waferSize_ + waferSepar_), nFine, nCoarse);
  for (unsigned int k = 0; k < tag_.size(); ++k) {
    // First the mother
    std::vector<double> xM = {rM, 0, -rM, -rM, 0, rM};
    std::vector<double> yM = {RM2, 2 * RM2, RM2, -RM2, -2 * RM2, -RM2};
    std::vector<double> zw = {-0.5 * thick_, 0.5 * thick_};
    std::vector<double> zx(2, 0), zy(2, 0), scale(2, 1.0);
    std::string parentName = parent().name().name();
    parentName = parentName + tag_[k] + waferTag_;
    DDSolid solid = DDSolidFactory::extrudedpolygon(parentName, xM, yM, zw, zx, zy, scale);
    DDName matName(DDSplit(material_).first, DDSplit(material_).second);
    DDMaterial matter(matName);
    DDLogicalPart glogM = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: " << solid.name() << " extruded polygon made of "
                                  << matName << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":"
                                  << scale[0] << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":"
                                  << scale[1] << " and " << xM.size() << " edges";
    for (unsigned int kk = 0; kk < xM.size(); ++kk)
      edm::LogVerbatim("HGCalGeom") << "[" << kk << "] " << xM[kk] << ":" << yM[kk];
#endif

    // Then the layers
    std::vector<double> xL = {r2, 0, -r2, -r2, 0, r2};
    std::vector<double> yL = {R2, 2 * R2, R2, -R2, -2 * R2, -R2};
    std::vector<DDLogicalPart> glogs(materials_.size());
    for (unsigned int ii = 0; ii < copyNumber_.size(); ii++) {
      copyNumber_[ii] = 1;
    }
    double zi(-0.5 * thick_), thickTot(0.0);
    for (unsigned int l = 0; l < layers_.size(); l++) {
      unsigned int i = layers_[l];
      if (copyNumber_[i] == 1) {
        if (layerType_[i] > 0) {
          zw[0] = -0.5 * waferThick_;
          zw[1] = 0.5 * waferThick_;
        } else {
          zw[0] = -0.5 * layerThick_[i];
          zw[1] = 0.5 * layerThick_[i];
        }
        std::string layerName = layerNames_[i] + tag_[k] + waferTag_;
        solid = DDSolidFactory::extrudedpolygon(layerName, xL, yL, zw, zx, zy, scale);
        DDName matN(DDSplit(materials_[i]).first, DDSplit(materials_[i]).second);
        DDMaterial matter(matN);
        glogs[i] = DDLogicalPart(solid.ddname(), matter, solid);

#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: " << solid.name() << " extruded polygon made of "
                                      << matN << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":"
                                      << scale[0] << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":"
                                      << scale[1] << " and " << xL.size() << " edges";
        for (unsigned int kk = 0; kk < xL.size(); ++kk)
          edm::LogVerbatim("HGCalGeom") << "[" << kk << "] " << xL[kk] << ":" << yL[kk];
#endif
      }
      DDTranslation tran0(0, 0, (zi + 0.5 * layerThick_[i]));
      DDRotation rot;
      cpv.position(glogs[i], glogM, copyNumber_[i], tran0, rot);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferFullRotated: " << glogs[i].name() << " number " << copyNumber_[i]
                                    << " positioned in " << glogM.name() << " at " << tran0 << " with no rotation";
#endif
      ++copyNumber_[i];
      zi += layerThick_[i];
      thickTot += layerThick_[i];
      if (layerType_[i] > 0) {
        //int n2 = nCells_ / 2;
        for (int u = 0; u < 2 * nCells_; ++u) {
          for (int v = 0; v < 2 * nCells_; ++v) {
            if (((v - u) < nCells_) && ((u - v) <= nCells_)) {
              int placeIndex = wafer.cellPlacementIndex(1, HGCalTypes::waferFrontBack(face_[k]), orient_[k]);
              std::pair<double, double> xy1 = wafer.cellUV2XY1(u, v, placeIndex, cellType_);
              double yp = xy1.second;
              double xp = xy1.first;
              int cell(0);
              std::pair<int, int> cell1 = wafer.cellUV2Cell(u, v, placeIndex, cellType_);
              cell = cell1.first + cellOffset_[cell1.second];
              DDTranslation tran(xp, yp, 0);
              int copy = HGCalTypes::packCellTypeUV(cellType_, u, v);
              cpv.position(DDName(cellNames_[cell]), glogs[i], copy, tran, rot);
#ifdef EDM_ML_DEBUG
              edm::LogVerbatim("HGCalGeom")
                  << "DDHGCalWaferFullRotated: " << cellNames_[cell] << " number " << copy << " positioned in "
                  << glogs[i].name() << " at " << tran << " with no rotation";
#endif
            }
          }
        }
      }
    }
    if (std::abs(thickTot - thick_) >= tol) {
      if (thickTot > thick_) {
        edm::LogError("HGCalGeom") << "Thickness of the partition " << thick_ << " is smaller than " << thickTot
                                   << ": thickness of all its components **** ERROR ****";
      } else {
        edm::LogWarning("HGCalGeom") << "Thickness of the partition " << thick_ << " does not match with " << thickTot
                                     << " of the components";
      }
    }
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalWaferFullRotated, "hgcal:DDHGCalWaferFullRotated");
