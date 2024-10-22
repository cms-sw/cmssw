///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalWaferF.cc
// Description: Geometry factory class for a full silicon Wafer
// Created by Sunanda Banerjee
// Extended for rotated wafer by Pruthvi Suryadevara
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

#include <string>
#include <vector>
#include <sstream>

//#define EDM_ML_DEBUG

class DDHGCalWaferF : public DDAlgorithm {
public:
  // Constructor and Destructor
  DDHGCalWaferF() {}
  ~DDHGCalWaferF() override = default;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

private:
  std::string material_;                 // Material name for module with gap
  double thick_;                         // Module thickness
  double waferSize_;                     // Wafer size
  double waferSepar_;                    // Sensor separation
  double waferThick_;                    // Wafer thickness
  std::vector<std::string> layerNames_;  // Names of the layers
  std::vector<std::string> materials_;   // Materials of the layers
  std::vector<double> layerThick_;       // Thickness of layers
  std::vector<int> layerType_;           // Layer types
  std::vector<int> copyNumber_;          // Initial copy numbers
  std::vector<int> layers_;              // Number of layers in a section
  int nCells_;                           // Half number of cells along u-v axis
  int cellType_;                         // Cell Type (0,1,2: Fine, Course 2/3)
  std::vector<std::string> cellNames_;   // Name of the cells
  std::string nameSpace_;                // Namespace to be used
};

void DDHGCalWaferF::initialize(const DDNumericArguments& nArgs,
                               const DDVectorArguments& vArgs,
                               const DDMapArguments&,
                               const DDStringArguments& sArgs,
                               const DDStringVectorArguments& vsArgs) {
  material_ = sArgs["ModuleMaterial"];
  thick_ = nArgs["ModuleThickness"];
  waferSize_ = nArgs["WaferSize"];
  waferSepar_ = nArgs["SensorSeparation"];
  waferThick_ = nArgs["WaferThickness"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferF: Module " << parent().name() << " made of " << material_ << " T "
                                << thick_ << " Wafer 2r " << waferSize_ << " Half Separation " << waferSepar_ << " T "
                                << waferThick_;
#endif
  layerNames_ = vsArgs["LayerNames"];
  materials_ = vsArgs["LayerMaterials"];
  layerThick_ = vArgs["LayerThickness"];
  layerType_ = dbl_to_int(vArgs["LayerTypes"]);
  copyNumber_.resize(materials_.size(), 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferF: " << layerNames_.size() << " types of volumes";
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
  nameSpace_ = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferF: Cells/Wafer " << nCells_ << " Cell Type " << cellType_
                                << " NameSpace " << nameSpace_ << ": # of cells " << cellNames_.size();
  for (unsigned int k = 0; k < cellNames_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferF: Cell[" << k << "] " << cellNames_[k];
#endif
}

void DDHGCalWaferF::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  int counter(0);
#endif

  static constexpr double tol = 0.00001;
  static const double sqrt3 = std::sqrt(3.0);
  double rM = 0.5 * waferSize_;
  double RM2 = rM / sqrt3;
  double R = waferSize_ / (3.0 * nCells_);
  double r = 0.5 * R * sqrt3;
  double r2 = 0.5 * waferSize_;
  double R2 = r2 / sqrt3;

  // First the mother
  std::vector<double> xM = {rM, 0, -rM, -rM, 0, rM};
  std::vector<double> yM = {RM2, 2 * RM2, RM2, -RM2, -2 * RM2, -RM2};
  std::vector<double> zw = {-0.5 * thick_, 0.5 * thick_};
  std::vector<double> zx(2, 0), zy(2, 0), scale(2, 1.0);
  DDName parentName = parent().name();
  DDSolid solid = DDSolidFactory::extrudedpolygon(parentName, xM, yM, zw, zx, zy, scale);
  DDName matName(DDSplit(material_).first, DDSplit(material_).second);
  DDMaterial matter(matName);
  DDLogicalPart glogM = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferF: " << solid.name() << " extruded polygon made of " << matName
                                << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                << " and " << xM.size() << " edges";
  for (unsigned int k = 0; k < xM.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xM[k] << ":" << yM[k];
#endif

  // Then the layers
  std::vector<double> xL = {r2, 0, -r2, -r2, 0, r2};
  std::vector<double> yL = {R2, 2 * R2, R2, -R2, -2 * R2, -R2};
  std::vector<DDLogicalPart> glogs(materials_.size());
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
      solid = DDSolidFactory::extrudedpolygon(layerNames_[i], xL, yL, zw, zx, zy, scale);
      DDName matN(DDSplit(materials_[i]).first, DDSplit(materials_[i]).second);
      DDMaterial matter(matN);
      glogs[i] = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferF: " << solid.name() << " extruded polygon made of " << matN
                                    << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                    << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                    << " and " << xL.size() << " edges";
      for (unsigned int k = 0; k < xL.size(); ++k)
        edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xL[k] << ":" << yL[k];
#endif
    }
    DDTranslation tran0(0, 0, (zi + 0.5 * layerThick_[i]));
    DDRotation rot;
    cpv.position(glogs[i], glogM, copyNumber_[i], tran0, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferF: " << glogs[i].name() << " number " << copyNumber_[i]
                                  << " positioned in " << glogM.name() << " at " << tran0 << " with no rotation";
#endif
    ++copyNumber_[i];
    zi += layerThick_[i];
    thickTot += layerThick_[i];
    if (layerType_[i] > 0) {
      int n2 = nCells_ / 2;
      double y0 = (cellType_ >= 3) ? 0.5 : 0.0;
      double x0 = (cellType_ >= 3) ? 0.5 : 1.0;
      int voff = (cellType_ >= 3) ? 0 : 1;
      int uoff = 1 - voff;
      int cellType = (cellType_ >= 3) ? (cellType_ - 3) : cellType_;
      for (int u = 0; u < 2 * nCells_; ++u) {
        for (int v = 0; v < 2 * nCells_; ++v) {
          if (((v - u) < (nCells_ + uoff)) && (u - v) < (nCells_ + voff)) {
#ifdef EDM_ML_DEBUG
            counter++;
#endif
            double yp = (u - 0.5 * v - n2 + y0) * 2 * r;
            double xp = (1.5 * (v - nCells_) + x0) * R;
            int cell(0);
            if ((u == 0) && (v == 0))
              cell = 7;
            else if ((u == 0) && (v == nCells_ - voff))
              cell = 8;
            else if ((u == nCells_ - uoff) && (v == 2 * nCells_ - 1))
              cell = 9;
            else if ((u == (2 * nCells_ - 1)) && (v == 2 * nCells_ - 1))
              cell = 10;
            else if ((u == 2 * nCells_ - 1) && (v == (nCells_ - voff)))
              cell = 11;
            else if ((u == (nCells_ - uoff)) && (v == 0))
              cell = 12;
            else if (u == 0)
              cell = 1;
            else if ((v - u) == (nCells_ - voff))
              cell = 4;
            else if (v == (2 * nCells_ - 1))
              cell = 2;
            else if (u == (2 * nCells_ - 1))
              cell = 5;
            else if ((u - v) == (nCells_ - uoff))
              cell = 3;
            else if (v == 0)
              cell = 6;
            if ((cellType_ >= 3) && (cell != 0))
              cell += 12;
            DDTranslation tran(xp, yp, 0);
            int copy = HGCalTypes::packCellTypeUV(cellType, u, v);
            cpv.position(DDName(cellNames_[cell]), glogs[i], copy, tran, rot);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom")
                << "DDHGCalWaferF: " << cellNames_[cell] << " number " << copy << " positioned in " << glogs[i].name()
                << " at " << tran << " with no rotation";
#endif
          }
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "\nDDHGCalWaferF::Counter : " << counter << "\n===============================\n";
#endif
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

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalWaferF, "hgcal:DDHGCalWaferF");
