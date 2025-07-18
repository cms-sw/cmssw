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

//#define EDM_ML_DEBUG

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
  std::string moduleMaterial_;              // Material name for mother volume
  double moduleThick_;                      // Thickness of the overall moduled
  int sectors_;                             // Number of phi sectors (cassettes)
  std::vector<std::string> tagsector_;      // Tags of the sectors
  int position_;                            // 0 if -z; 1 if +z;
  std::vector<std::string> tagpos_;         // Tags for the modules
  std::vector<int> xsignpos_;               // sign of the x-value;
  double phi0_;                             // Start phi of the first cassette
  double dphi_;                             // Delta phi between cassettes
  std::vector<std::string> absNames_;       // Names of the absorber layers
  std::vector<int> absN_;                   // Number of point in each layer
  std::vector<double> absX_;                // x coordinates of abs layers
  std::vector<double> absY_;                // x coordinates of abs layers
  std::vector<std::string> layerNames_;     // Names of layers within abs layer
  std::vector<std::string> layerMaterial_;  // Materials of the layers
  std::vector<double> layerThick_;          // Thickness of layers
  std::vector<int> layerType_;              // Layer types
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
  moduleMaterial_ = sArgs["ModuleMaterial"];
  moduleThick_ = nArgs["ModuleThick"];
  sectors_ = static_cast<int>(nArgs["Sectors"]);
  position_ = static_cast<int>(nArgs["Position"]);
  phi0_ = convertDegToRad(nArgs["PhiStart"]);
  dphi_ = (2._pi) / sectors_;
  if (position_ == 0) {
    tagpos_.emplace_back("PN");
    xsignpos_.emplace_back(-1);
  } else {
    tagpos_.emplace_back("PP");
    xsignpos_.emplace_back(1);
  }
  for (int k = 0; k < sectors_; ++k)
    tagsector_.emplace_back("F" + std::to_string(k));
#ifdef EDM_ML_DEBUG
  std::ostringstream st0, st1;
  for (unsigned int k = 0; k < tagsector_.size(); ++k)
    st0 << ": " << tagsector_[k];
  for (unsigned int k = 0; k < tagpos_.size(); ++k)
    st1 << " " << tagpos_[k] << ":" << xsignpos_[k];
  edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: " << tagpos_.size() << " Modules with base name " << parent().name()
                                << " made of " << moduleMaterial_ << " T " << moduleThick_ << " having " << sectors_
                                << " sectors" << st0.str() << " phi0 " << convertRadToDeg(phi0_) << " dphi "
                                << convertRadToDeg(dphi_) << " Tags:" << st1.str();
#endif

  layerNames_ = vsArgs["LayerNames"];
  layerMaterial_ = vsArgs["LayerMaterials"];
  layerThick_ = vArgs["LayerThickness"];
  layerType_ = dbl_to_int(vArgs["LayerType"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalPassive: " << layerNames_.size() << " types of volumes";
  for (unsigned int i = 0; i < layerNames_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << layerNames_[i] << " of thickness " << layerThick_[i]
                                  << " filled with " << layerMaterial_[i];
  std::ostringstream st2;
  for (unsigned int i = 0; i < layerType_.size(); ++i)
    st2 << " [" << i << "] " << layerType_[i];
  edm::LogVerbatim("HGCalGeom") << "There are " << layerType_.size() << " blocks" << st2.str();
#endif

  absNames_ = vsArgs["AbsorberName"];
  absN_ = dbl_to_int(vArgs["AbsorberN"]);
  absX_ = vArgs["AbsorberX"];
  absY_ = vArgs["AbsorberY"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "There are " << absNames_.size() << " basic absorber shapes:";
  unsigned int j(0);
  for (unsigned int k = 0; k < absNames_.size(); ++k) {
    std::ostringstream st3;
    st3 << absNames_[k] << " with " << absN_[k] << " points:";
    for (int i = 0; i < absN_[k]; ++i)
      st3 << " (" << absX_[j + i] << ", " << absY_[j + i] << ")";
    j += absN_[k];
    edm::LogVerbatim("HGCalGeom") << st3.str();
  }
#endif
}

void DDHGCalPassive::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Executing DDHGCalPassive...";
#endif
  static constexpr double tol = 0.00001;

  // Loop over positions
  for (unsigned int i1 = 0; i1 < tagpos_.size(); ++i1) {
    // Loop over sectors
    for (int i2 = 0; i2 < sectors_; ++i2) {
      double phi = phi0_ + i2 * dphi_;
      double cphi = std::cos(phi);
      double sphi = std::sin(phi);
      // Loop over passive volumes
      int j(0);
      for (unsigned i3 = 0; i3 < absNames_.size(); ++i3) {
        //First make the mother
        std::string parentName = parent().name().name() + absNames_[i3] + tagsector_[i2] + tagpos_[i1];
        std::vector<double> zw = {-0.5 * moduleThick_, 0.5 * moduleThick_};
        std::vector<double> zx(2, 0), zy(2, 0), scale(2, 1.0);
        std::vector<double> xM(absN_[i3], 0), yM(absN_[i3], 0);
        for (int k = 0; k < absN_[i3]; ++k) {
          xM[k] = xsignpos_[i1] * (cphi * absX_[j + k] + sphi * absY_[j + k]);
          yM[k] = -sphi * absX_[j + k] + cphi * absY_[j + k];
        }
        j += absN_[i3];
        DDSolid solid = DDSolidFactory::extrudedpolygon(parentName, xM, yM, zw, zx, zy, scale);
        DDName matName(DDSplit(moduleMaterial_).first, DDSplit(moduleMaterial_).second);
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
        std::vector<DDLogicalPart> glogs(layerMaterial_.size());
        std::vector<int> copyNumber(layerMaterial_.size(), 1);
        double zi(-0.5 * moduleThick_), thickTot(0.0);
        for (unsigned int l = 0; l < layerType_.size(); l++) {
          unsigned int i = layerType_[l];
          if (copyNumber[i] == 1) {
            zw[0] = -0.5 * layerThick_[i];
            zw[1] = 0.5 * layerThick_[i];
            std::string layerName = parentName + layerNames_[i];
            solid = DDSolidFactory::extrudedpolygon(layerName, xM, yM, zw, zx, zy, scale);
            DDName matN(DDSplit(layerMaterial_[i]).first, DDSplit(layerMaterial_[i]).second);
            DDMaterial matter(matN);
            glogs[i] = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom")
                << "DDHGCalPassive: Layer " << i << ":" << l << ":" << solid.name() << " extruded polygon made of "
                << matN << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1] << " and " << xM.size()
                << " edges";
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
        if ((std::abs(thickTot - moduleThick_) >= tol) && (!layerType_.empty())) {
          if (thickTot > moduleThick_) {
            edm::LogError("HGCalGeom") << "Thickness of the partition " << moduleThick_ << " is smaller than "
                                       << thickTot << ": thickness of all its components **** ERROR ****";
          } else {
            edm::LogWarning("HGCalGeom") << "Thickness of the partition " << moduleThick_ << " does not match with "
                                         << thickTot << " of the components";
          }
        }
      }
    }
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalPassive, "hgcal:DDHGCalPassive");
