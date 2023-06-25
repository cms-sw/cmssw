///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalPassiveFull.cc
// Description: Geometry factory class for a full silicon Wafer
// Created by Sunanda Banerjee
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

#include <string>
#include <vector>
#include <sstream>

//#define EDM_ML_DEBUG

class DDHGCalPassiveFull : public DDAlgorithm {
public:
  // Constructor and Destructor
  DDHGCalPassiveFull();
  ~DDHGCalPassiveFull() override = default;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

private:
  std::string material_;                 // Material name for module
  double thick_;                         // Module thickness
  double waferSize_;                     // Wafer size
  double waferSepar_;                    // Sensor separation
  std::vector<std::string> layerNames_;  // Names of the layers
  std::vector<std::string> materials_;   // Materials of the layers
  std::vector<double> layerThick_;       // Thickness of layers
  std::vector<int> copyNumber_;          // Initial copy numbers
  std::vector<int> layerType_;           // Layer types
};

DDHGCalPassiveFull::DDHGCalPassiveFull() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalPassiveFull: Creating an instance";
#endif
}

void DDHGCalPassiveFull::initialize(const DDNumericArguments& nArgs,
                                    const DDVectorArguments& vArgs,
                                    const DDMapArguments&,
                                    const DDStringArguments& sArgs,
                                    const DDStringVectorArguments& vsArgs) {
  material_ = sArgs["ModuleMaterial"];
  thick_ = nArgs["ModuleThickness"];
  waferSize_ = nArgs["WaferSize"];
#ifdef EDM_ML_DEBUG
  waferSepar_ = nArgs["SensorSeparation"];
  edm::LogVerbatim("HGCalGeom") << "DDHGCalPassiveFull: Module " << parent().name() << " made of " << material_ << " T "
                                << thick_ << " Wafer 2r " << waferSize_ << " Half Separation " << waferSepar_;
#endif
  layerNames_ = vsArgs["LayerNames"];
  materials_ = vsArgs["LayerMaterials"];
  layerThick_ = vArgs["LayerThickness"];
  copyNumber_.resize(materials_.size(), 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalPassiveFull: " << layerNames_.size() << " types of volumes";
  for (unsigned int i = 0; i < layerNames_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << layerNames_[i] << " of thickness " << layerThick_[i]
                                  << " filled with " << materials_[i];
#endif
  layerType_ = dbl_to_int(vArgs["LayerType"]);
#ifdef EDM_ML_DEBUG
  std::ostringstream st1;
  for (unsigned int i = 0; i < layerType_.size(); ++i)
    st1 << " [" << i << "] " << layerType_[i];
  edm::LogVerbatim("HGCalGeom") << "There are " << layerType_.size() << " blocks" << st1.str();
#endif
}

void DDHGCalPassiveFull::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Executing DDHGCalPassiveFull...";
#endif

  static constexpr double tol = 0.00001;
  static const double sqrt3 = std::sqrt(3.0);
  double rM = 0.5 * waferSize_;
  double RM2 = rM / sqrt3;

  // First the mother
  std::vector<double> xM = {rM, 0, -rM, -rM, 0, rM};
  std::vector<double> yM = {RM2, 2 * RM2, RM2, -RM2, -2 * RM2, -RM2};
  std::vector<double> zw = {-0.5 * thick_, 0.5 * thick_};
  std::vector<double> zx(2, 0), zy(2, 0), scale(2, 1.0);
  std::string parentName = parent().name().name();
  DDSolid solid = DDSolidFactory::extrudedpolygon(parentName, xM, yM, zw, zx, zy, scale);
  DDName matName(DDSplit(material_).first, DDSplit(material_).second);
  DDMaterial matter(matName);
  DDLogicalPart glogM = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalPassiveFull: " << solid.name() << " extruded polygon made of " << matName
                                << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                << " and " << xM.size() << " edges";
  for (unsigned int kk = 0; kk < xM.size(); ++kk)
    edm::LogVerbatim("HGCalGeom") << "[" << kk << "] " << xM[kk] << ":" << yM[kk];
#endif

  // Then the layers
  std::vector<DDLogicalPart> glogs(materials_.size());
  double zi(-0.5 * thick_), thickTot(0.0);
  for (unsigned int l = 0; l < layerType_.size(); l++) {
    unsigned int i = layerType_[l];
    if (copyNumber_[i] == 1) {
      zw[0] = -0.5 * layerThick_[i];
      zw[1] = 0.5 * layerThick_[i];
      std::string layerName = parentName + layerNames_[i];
      solid = DDSolidFactory::extrudedpolygon(layerName, xM, yM, zw, zx, zy, scale);
      DDName matN(DDSplit(materials_[i]).first, DDSplit(materials_[i]).second);
      DDMaterial matter(matN);
      glogs[i] = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalPassiveFull: Layer " << i << ":" << l << ":" << solid.name()
                                    << " extruded polygon made of " << matN << " z|x|y|s (0) " << zw[0] << ":" << zx[0]
                                    << ":" << zy[0] << ":" << scale[0] << " z|x|y|s (1) " << zw[1] << ":" << zx[1]
                                    << ":" << zy[1] << ":" << scale[1] << " and " << xM.size() << " edges";
      for (unsigned int kk = 0; kk < xM.size(); ++kk)
        edm::LogVerbatim("HGCalGeom") << "[" << kk << "] " << xM[kk] << ":" << yM[kk];
#endif
    }
    DDTranslation tran0(0, 0, (zi + 0.5 * layerThick_[i]));
    DDRotation rot;
    cpv.position(glogs[i], glogM, copyNumber_[i], tran0, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalPassiveFull: " << glogs[i].name() << " number " << copyNumber_[i]
                                  << " positioned in " << glogM.name() << " at " << tran0 << " with no rotation";
#endif
    ++copyNumber_[i];
    zi += layerThick_[i];
    thickTot += layerThick_[i];
  }
  if ((std::abs(thickTot - thick_) >= tol) && (!layerType_.empty())) {
    if (thickTot > thick_) {
      edm::LogError("HGCalGeom") << "Thickness of the partition " << thick_ << " is smaller than " << thickTot
                                 << ": thickness of all its components **** ERROR ****";
    } else {
      edm::LogWarning("HGCalGeom") << "Thickness of the partition " << thick_ << " does not match with " << thickTot
                                   << " of the components";
    }
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalPassiveFull, "hgcal:DDHGCalPassiveFull");
