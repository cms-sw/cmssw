///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalPassive.cc
// Description: Makes layers of passive in the size of cassette
// Created by Sunanda Banerjee
///////////////////////////////////////////////////////////////////////////////
#include "DataFormats/Math/interface/angle_units.h"
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
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"

#include <string>
#include <vector>
#include <sstream>

#define EDM_ML_DEBUG
using namespace angle_units::operators;

class DDHGCalPassive : public DDAlgorithm {
public:
  // Constructor and Destructor
  DDHGCalPassive();
  ~DDHGCalPassive() override = default;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

private:
  std::string material_;                 // Material name for mother volume
  double thick_;                         // Thickness of the section
  double zMinBlock_;                     // z-position of the first layer
  double moduleThick_;                   // Thickness of the overall module
  std::vector<std::string> tagLayer_;    // Tag of the layer (to be added to name)
  int sectors_;                          // Number of phi sectors (cassettes)
  int parts_;                            // number of parts in units of 30 degree
  std::vector<std::string> tagSector_;   // Tag of the sector (to be added to name)
  double phi0_;                          // Start phi of the first cassette
  double dphi_;                          // delta phi of the cassette
  double shiftTop_;                      // Tolerance at the top
  double shiftBot_;                      // Tolerance at the bottom
  std::vector<std::string> layerNames_;  // Names of the layers
  std::vector<std::string> materials_;   // Materials of the layers
  std::vector<double> layerThick_;       // Thickness of layers
  std::vector<int> layerType_;           // Layer types
  std::vector<double> slopeB_;           // Slope at the lower R
  std::vector<double> zFrontB_;          // Starting Z values for the slopes
  std::vector<double> rMinFront_;        // Corresponding rMin's
  std::vector<double> slopeT_;           // Slopes at the larger R
  std::vector<double> zFrontT_;          // Starting Z values for the slopes
  std::vector<double> rMaxFront_;        // Corresponding rMax's
};

DDHGCalPassive::DDHGCalPassive() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: Creating an instance";
#endif
}

void DDHGCalPassive::initialize(const DDNumericArguments& nArgs,
                                const DDVectorArguments& vArgs,
                                const DDMapArguments&,
                                const DDStringArguments& sArgs,
                                const DDStringVectorArguments& vsArgs) {
  material_ = sArgs["ModuleMaterial"];
  thick_ = nArgs["Thickness"];
  zMinBlock_ = nArgs["zMinBlock"];
  moduleThick_ = nArgs["ModuleThick"];
  tagLayer_ = vsArgs["TagLayer"];
  tagSector_ = vsArgs["TagSector"];
  parts_ = static_cast<int>(nArgs["Parts"]);
  phi0_ = convertDegToRad(nArgs["PhiStart"]);
  dphi_ = (2._pi) / tagSector_.size();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: " << tagLayer_.size() << " Modules with base name "
                                << parent().name() << " made of " << material_ << " T " << thick_ << " Sectors "
                                << tagSector_.size() << " Parts " << parts_ << " phi0 " << convertRadToDeg(phi0_);
  for (unsigned int i = 0; i < tagLayer_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Layer " << i << " Tag " << tagLayer_[i] << " T " << moduleThick_;
  for (unsigned int i = 0; i < tagSector_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Sector " << i << " Tag " << tagSector_[i] << " W " << convertRadToDeg(dphi_);
#endif

  layerNames_ = vsArgs["LayerNames"];
  materials_ = vsArgs["LayerMaterials"];
  layerThick_ = vArgs["LayerThickness"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: " << layerNames_.size() << " types of volumes";
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

  shiftTop_ = nArgs["ShiftTop"];
  shiftBot_ = nArgs["ShiftBottom"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Shifts st the top " << shiftTop_ << " and at the bottom " << shiftBot_;
#endif
  slopeB_ = vArgs["SlopeBottom"];
  zFrontB_ = vArgs["ZFrontBottom"];
  rMinFront_ = vArgs["RMinFront"];
  slopeT_ = vArgs["SlopeTop"];
  zFrontT_ = vArgs["ZFrontTop"];
  rMaxFront_ = vArgs["RMaxFront"];
#ifdef EDM_ML_DEBUG
  for (unsigned int i = 0; i < slopeB_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Bottom Block [" << i << "] Zmin " << zFrontB_[i] << " Rmin " << rMinFront_[i]
                                  << " Slope " << slopeB_[i];
  for (unsigned int i = 0; i < slopeT_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Top Block [" << i << "] Zmin " << zFrontT_[i] << " Rmax " << rMaxFront_[i]
                                  << " Slope " << slopeT_[i];
#endif
}

void DDHGCalPassive::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Executing DDHGCalPassive...";
#endif

  static constexpr double tol = 0.00001;

  // Loop over Layers
  double zim(zMinBlock_);
  //Loop over layers
  for (unsigned int j = 0; j < tagLayer_.size(); ++j) {
    double routF = HGCalGeomTools::radius(zim, zFrontT_, rMaxFront_, slopeT_) - shiftTop_;
    double zo = zim + moduleThick_;
    double rinB = HGCalGeomTools::radius(zo, zFrontB_, rMinFront_, slopeB_) + shiftBot_;
    zim += moduleThick_;
    for (unsigned int k = 0; k < tagSector_.size(); ++k) {
      std::string parentName = parent().name().name() + tagLayer_[j] + tagSector_[k];
      double phi1 = phi0_ + k * dphi_;
      double phi2 = phi1 + dphi_;
      double phi0 = phi1 + 0.5 * dphi_;
      // First the mother
      std::vector<double> xM, yM;
      if (parts_ == 1) {
        xM = {rinB * cos(phi1), routF * cos(phi1), routF * cos(phi2), rinB * cos(phi2)};
        yM = {rinB * sin(phi1), routF * sin(phi1), routF * sin(phi2), rinB * sin(phi2)};
      } else {
        xM = {rinB * cos(phi1),
              routF * cos(phi1),
              routF * cos(phi0),
              routF * cos(phi2),
              rinB * cos(phi2),
              rinB * cos(phi0)};
        yM = {rinB * sin(phi1),
              routF * sin(phi1),
              routF * sin(phi0),
              routF * sin(phi2),
              rinB * sin(phi2),
              rinB * sin(phi0)};
      }
      std::vector<double> zw = {-0.5 * thick_, 0.5 * thick_};
      std::vector<double> zx(2, 0), zy(2, 0), scale(2, 1.0);
      DDSolid solid = DDSolidFactory::extrudedpolygon(parentName, xM, yM, zw, zx, zy, scale);
      DDName matName(DDSplit(material_).first, DDSplit(material_).second);
      DDMaterial matter(matName);
      DDLogicalPart glogM = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: " << solid.name() << " extruded polygon made of " << matName
                                    << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                    << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                    << " and " << xM.size() << " edges";
      for (unsigned int kk = 0; kk < xM.size(); ++kk)
        edm::LogVerbatim("HGCalGeom") << "[" << kk << "] " << xM[kk] << ":" << yM[kk];
#endif

      // Then the layers
      std::vector<DDLogicalPart> glogs(materials_.size());
      std::vector<int> copyNumber(materials_.size(), 1);
      double zi(-0.5 * thick_), thickTot(0.0);
      for (unsigned int l = 0; l < layerType_.size(); l++) {
        unsigned int i = layerType_[l];
        if (copyNumber[i] == 1) {
          zw[0] = -0.5 * layerThick_[i];
          zw[1] = 0.5 * layerThick_[i];
          std::string layerName = parentName + layerNames_[i];
          solid = DDSolidFactory::extrudedpolygon(layerName, xM, yM, zw, zx, zy, scale);
          DDName matN(DDSplit(materials_[i]).first, DDSplit(materials_[i]).second);
          DDMaterial matter(matN);
          glogs[i] = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: Layer " << i << ":" << l << ":" << solid.name()
                                        << " extruded polygon made of " << matN << " z|x|y|s (0) " << zw[0] << ":"
                                        << zx[0] << ":" << zy[0] << ":" << scale[0] << " z|x|y|s (1) " << zw[1] << ":"
                                        << zx[1] << ":" << zy[1] << ":" << scale[1] << " and " << xM.size() << " edges";
          for (unsigned int kk = 0; kk < xM.size(); ++kk)
            edm::LogVerbatim("HGCalGeom") << "[" << kk << "] " << xM[kk] << ":" << yM[kk];
#endif
        }
        DDTranslation tran0(0, 0, (zi + 0.5 * layerThick_[i]));
        DDRotation rot;
        cpv.position(glogs[i], glogM, copyNumber[i], tran0, rot);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: " << glogs[i].name() << " number " << copyNumber[i]
                                      << " positioned in " << glogM.name() << " at " << tran0 << " with no rotation";
#endif
        ++copyNumber[i];
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
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalPassive, "hgcal:DDHGCalPassive");
