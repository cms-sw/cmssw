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
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"

#include <string>
#include <vector>
#include <sstream>

//#define EDM_ML_DEBUG

class DDHGCalWaferP : public DDAlgorithm {
public:
  // Constructor and Destructor
  DDHGCalWaferP() {}
  ~DDHGCalWaferP() override = default;

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
  std::vector<std::string> tags_;        // Tags to be added to each name
  std::vector<int> partialTypes_;        // Type of partial wafer
  std::vector<int> orientations_;        // Orientations of the wafers
  std::vector<std::string> layerNames_;  // Names of the layers
  std::vector<std::string> materials_;   // Materials of the layers
  std::vector<double> layerThick_;       // Thickness of layers
  std::vector<int> layerType_;           // Layer types
  std::vector<int> layers_;              // Number of layers in a section
  std::string senseName_;                // Name of the sensitive layer
  double senseT_;                        // Thickness of sensitive layer
  int senseType_;                        // Cell Type (0,1,2: Fine, Course 2/3)
  int posSense_;                         // Position depleted layer
  std::string nameSpace_;                // Namespace to be used
};

void DDHGCalWaferP::initialize(const DDNumericArguments& nArgs,
                               const DDVectorArguments& vArgs,
                               const DDMapArguments&,
                               const DDStringArguments& sArgs,
                               const DDStringVectorArguments& vsArgs) {
  material_ = sArgs["ModuleMaterial"];
  thick_ = nArgs["ModuleThickness"];
  waferSize_ = nArgs["WaferSize"];
  waferThick_ = nArgs["WaferThickness"];
#ifdef EDM_ML_DEBUG
  waferSepar_ = nArgs["SensorSeparation"];
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferP: Module " << parent().name() << " made of " << material_ << " T "
                                << thick_ << " Wafer 2r " << waferSize_ << " Half Separation " << waferSepar_ << " T "
                                << waferThick_;
#endif
  tags_ = vsArgs["Tags"];
  partialTypes_ = dbl_to_int(vArgs["PartialTypes"]);
  orientations_ = dbl_to_int(vArgs["Orientations"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferP: " << tags_.size() << " variations of wafer types";
  for (unsigned int k = 0; k < tags_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "Type[" << k << "] " << tags_[k] << " Partial " << partialTypes_[k]
                                  << " Orientation " << orientations_[k];
#endif
  layerNames_ = vsArgs["LayerNames"];
  materials_ = vsArgs["LayerMaterials"];
  layerThick_ = vArgs["LayerThickness"];
  layerType_ = dbl_to_int(vArgs["LayerTypes"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferP: " << layerNames_.size() << " types of volumes";
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
  senseName_ = sArgs["SenseName"];
  senseT_ = nArgs["SenseThick"];
  senseType_ = static_cast<int>(nArgs["SenseType"]);
  posSense_ = static_cast<int>(nArgs["PosSensitive"]);
  nameSpace_ = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferP: NameSpace " << nameSpace_ << ": Sensitive Layer Name " << senseName_
                                << " Thickness " << senseT_ << " Type " << senseType_ << " Position " << posSense_;
#endif
}

void DDHGCalWaferP::execute(DDCompactView& cpv) {
  static constexpr double tol = 0.00001;
  std::string parentName = parent().name().name();

  // Loop over all types
  for (unsigned int k = 0; k < tags_.size(); ++k) {
    // First the mother
    std::string mother = parentName + tags_[k];
    std::vector<std::pair<double, double> > wxy =
        HGCalWaferMask::waferXY(partialTypes_[k], orientations_[k], 1, waferSize_, 0.0, 0.0, 0.0);
    std::vector<double> xM, yM;
    for (unsigned int i = 0; i < (wxy.size() - 1); ++i) {
      xM.emplace_back(wxy[i].first);
      yM.emplace_back(wxy[i].second);
    }
    std::vector<double> zw = {-0.5 * thick_, 0.5 * thick_};
    std::vector<double> zx(2, 0), zy(2, 0), scale(2, 1.0);
    DDSolid solid = DDSolidFactory::extrudedpolygon(mother, xM, yM, zw, zx, zy, scale);
    DDName matName(DDSplit(material_).first, DDSplit(material_).second);
    DDMaterial matter(matName);
    DDLogicalPart glogM = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferP: " << solid.name() << " extruded polygon made of " << matName
                                  << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                  << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                  << " partial " << partialTypes_[k] << " orientation " << orientations_[k] << " and "
                                  << xM.size() << " edges";
    for (unsigned int j = 0; j < xM.size(); ++j)
      edm::LogVerbatim("HGCalGeom") << "[" << j << "] " << xM[j] << ":" << yM[j];
#endif

    // Then the layers
    wxy = HGCalWaferMask::waferXY(partialTypes_[k], orientations_[k], 1, waferSize_, 0.0, 0.0, 0.0);
    std::vector<double> xL, yL;
    for (unsigned int i = 0; i < (wxy.size() - 1); ++i) {
      xL.emplace_back(wxy[i].first);
      yL.emplace_back(wxy[i].second);
    }
    std::vector<DDLogicalPart> glogs(materials_.size());
    std::vector<int> copyNumber(materials_.size(), 1);
    double zi(-0.5 * thick_), thickTot(0.0);
    for (unsigned int l = 0; l < layers_.size(); l++) {
      unsigned int i = layers_[l];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferP:Layer " << l << ":" << i << " T " << layerThick_[i] << " Copy "
                                    << copyNumber[i];
#endif
      DDRotation rot;
      if (copyNumber[i] == 1) {
        if (layerType_[i] > 0) {
          zw[0] = -0.5 * waferThick_;
          zw[1] = 0.5 * waferThick_;
        } else {
          zw[0] = -0.5 * layerThick_[i];
          zw[1] = 0.5 * layerThick_[i];
        }
        std::string lname = layerNames_[i] + tags_[k];
        solid = DDSolidFactory::extrudedpolygon(lname, xL, yL, zw, zx, zy, scale);
        DDName matN(DDSplit(materials_[i]).first, DDSplit(materials_[i]).second);
        DDMaterial matter(matN);
        glogs[i] = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferP: " << solid.name() << " extruded polygon made of " << matN
                                      << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                      << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                      << " partial " << partialTypes_[k] << " orientation " << orientations_[k]
                                      << " and " << xL.size() << " edges";
        for (unsigned int j = 0; j < xL.size(); ++j)
          edm::LogVerbatim("HGCalGeom") << "[" << j << "] " << xL[j] << ":" << yL[j];
#endif
        if (layerType_[i] > 0) {
          std::string sname = senseName_ + tags_[k];
          zw[0] = -0.5 * senseT_;
          zw[1] = 0.5 * senseT_;
          solid = DDSolidFactory::extrudedpolygon(sname, xL, yL, zw, zx, zy, scale);
          DDLogicalPart glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferP: " << solid.name() << " extruded polygon made of " << matN
                                        << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                        << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                        << " partial " << partialTypes_[k] << " orientation " << orientations_[k]
                                        << " and " << xL.size() << " edges";
          for (unsigned int j = 0; j < xL.size(); ++j)
            edm::LogVerbatim("HGCalGeom") << "[" << j << "] " << xL[j] << ":" << yL[j];
#endif
          double zpos = (posSense_ == 0) ? -0.5 * (waferThick_ - senseT_) : 0.5 * (waferThick_ - senseT_);
          DDTranslation tran(0, 0, zpos);
          int copy = 10 + senseType_;
          cpv.position(glog, glogs[i], copy, tran, rot);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferP: " << glog.name() << " number " << copy << " positioned in "
                                        << glogs[i].name() << " at " << tran << " with no rotation";
#endif
        }
      }
      DDTranslation tran0(0, 0, (zi + 0.5 * layerThick_[i]));
      cpv.position(glogs[i], glogM, copyNumber[i], tran0, rot);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferP: " << glogs[i].name() << " number " << copyNumber[i]
                                    << " positioned in " << glogM.name() << " at " << tran0 << " with no rotation";
#endif
      ++copyNumber[i];
      zi += layerThick_[i];
      thickTot += layerThick_[i];
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

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalWaferP, "hgcal:DDHGCalWaferP");
