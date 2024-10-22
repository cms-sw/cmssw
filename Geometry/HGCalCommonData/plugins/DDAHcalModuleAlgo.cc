///////////////////////////////////////////////////////////////////////////////
// File: DDAHcalModuleAlgo.cc
// Description: Geometry factory class for HGCal (EE and HESil)
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

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
#include "Geometry/HGCalCommonData/interface/AHCalParameters.h"

//#define EDM_ML_DEBUG

class DDAHcalModuleAlgo : public DDAlgorithm {
public:
  // Constructor and Destructor
  DDAHcalModuleAlgo();  // const std::string & name);

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

protected:
  void constructLayers(const DDLogicalPart&, DDCompactView& cpv);
  void positionSensitive(DDLogicalPart& glog, DDCompactView& cpv);

private:
  static constexpr double tol_ = 0.00001;

  std::string tile_;                    // Scintillator tile
  std::vector<std::string> materials_;  // Materials
  std::vector<std::string> names_;      // Names
  std::vector<double> thick_;           // Thickness of the material
  std::vector<int> copyNumber_;         // Initial copy numbers
  std::vector<int> layers_;             // Number of layers in a section
  std::vector<double> layerThick_;      // Thickness of each section
  std::vector<int> layerType_;          // Type of the layer
  std::vector<int> layerSense_;         // Content of a layer (sensitive?)
  std::vector<double> widths_;          // Width (passive, active)
  std::vector<double> heights_;         // Heights (passive, active)
  std::vector<int> tileN_;              // # of tiles (along x, y)
  std::vector<double> tileStep_;        // Separation between tiles (x, y)
  double zMinBlock_;                    // Starting z-value of the block
  std::string idNameSpace_;             // Namespace of this and ALL sub-parts
};

DDAHcalModuleAlgo::DDAHcalModuleAlgo() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: Creating an instance";
#endif
}

void DDAHcalModuleAlgo::initialize(const DDNumericArguments& nArgs,
                                   const DDVectorArguments& vArgs,
                                   const DDMapArguments&,
                                   const DDStringArguments& sArgs,
                                   const DDStringVectorArguments& vsArgs) {
  tile_ = sArgs["TileName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: Tile " << tile_;
#endif
  materials_ = vsArgs["MaterialNames"];
  names_ = vsArgs["VolumeNames"];
  thick_ = vArgs["Thickness"];
  copyNumber_.resize(materials_.size(), 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << materials_.size() << " types of volumes";
  for (unsigned int i = 0; i < names_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << names_[i] << " of thickness " << thick_[i]
                                  << " filled with " << materials_[i] << " first copy number " << copyNumber_[i];
#endif
  layers_ = dbl_to_int(vArgs["Layers"]);
  layerThick_ = vArgs["LayerThick"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << layers_.size() << " blocks";
  for (unsigned int i = 0; i < layers_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << layerThick_[i] << " with " << layers_[i]
                                  << " layers";
#endif
  layerType_ = dbl_to_int(vArgs["LayerType"]);
  layerSense_ = dbl_to_int(vArgs["LayerSense"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << layerType_.size() << " layers";
  for (unsigned int i = 0; i < layerType_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerType_[i] << " sensitive class "
                                  << layerSense_[i];
#endif
  widths_ = vArgs["Widths"];
  heights_ = vArgs["Heights"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << widths_.size() << " sizes for width "
                                << "and height:";
  for (unsigned int i = 0; i < widths_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << " [" << i << "] " << widths_[i] << ":" << heights_[i];
#endif
  tileN_ = dbl_to_int(vArgs["TileN"]);
  tileStep_ = vArgs["TileStep"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << tileN_.size() << " tile positioning parameters";
  for (unsigned int i = 0; i < tileN_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << " [" << i << "] " << tileN_[i] << ":" << tileStep_[i];
#endif
  zMinBlock_ = nArgs["zMinBlock"];
  idNameSpace_ = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalModule: zStart " << zMinBlock_ << "  NameSpace " << idNameSpace_;
#endif
}

////////////////////////////////////////////////////////////////////
// DDAHcalModuleAlgo methods...
////////////////////////////////////////////////////////////////////

void DDAHcalModuleAlgo::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDAHcalModuleAlgo...";
#endif
  constructLayers(parent(), cpv);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "<<== End of DDAHcalModuleAlgo construction";
#endif
}

void DDAHcalModuleAlgo::constructLayers(const DDLogicalPart& module, DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo test: \t\tInside Layers";
#endif
  double zi(zMinBlock_);
  int laymin(0);
  for (unsigned int i = 0; i < layers_.size(); i++) {
    double zo = zi + layerThick_[i];
    int laymax = laymin + layers_[i];
    double zz = zi;
    double thickTot(0);
    for (int ly = laymin; ly < laymax; ++ly) {
      int ii = layerType_[ly];
      int copy = copyNumber_[ii];
      zz += (0.5 * thick_[ii]);
      thickTot += thick_[ii];

      std::string name = "HGCal" + names_[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo test: Layer " << ly << ":" << ii << " Front " << zi
                                    << " Back " << zo << " superlayer thickness " << layerThick_[i];
#endif
      DDName matName(DDSplit(materials_[ii]).first, DDSplit(materials_[ii]).second);
      DDMaterial matter(matName);
      DDLogicalPart glog;
      if (layerSense_[ly] == 0) {
        DDSolid solid =
            DDSolidFactory::box(DDName(name, idNameSpace_), 0.5 * widths_[0], 0.5 * heights_[0], 0.5 * thick_[ii]);
        glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << solid.name() << " Box made of " << matName
                                      << " of dimensions " << 0.5 * widths_[0] << ", " << 0.5 * heights_[0] << ", "
                                      << 0.5 * thick_[ii];
#endif
      } else {
        DDSolid solid =
            DDSolidFactory::box(DDName(name, idNameSpace_), 0.5 * widths_[1], 0.5 * heights_[1], 0.5 * thick_[ii]);
        glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << solid.name() << " Box made of " << matName
                                      << " of dimensions " << 0.5 * widths_[1] << ", " << 0.5 * heights_[1] << ", "
                                      << 0.5 * thick_[ii];
#endif
        positionSensitive(glog, cpv);
      }
      DDTranslation r1(0, 0, zz);
      DDRotation rot;
      cpv.position(glog, module, copy, r1, rot);
      ++copyNumber_[ii];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << glog.name() << " number " << copy << " positioned in "
                                    << module.name() << " at " << r1 << " with " << rot;
#endif
      zz += (0.5 * thick_[ii]);
    }  // End of loop over layers in a block
    zi = zo;
    laymin = laymax;
    if (fabs(thickTot - layerThick_[i]) > tol_) {
      if (thickTot > layerThick_[i]) {
        edm::LogError("HGCalGeom") << "Thickness of the partition " << layerThick_[i] << " is smaller than thickness "
                                   << thickTot << " of all its components **** ERROR ****\n";
      } else {
        edm::LogWarning("HGCalGeom") << "Thickness of the partition " << layerThick_[i] << " does not match with "
                                     << thickTot << " of the components\n";
      }
    }
  }  // End of loop over blocks
}

void DDAHcalModuleAlgo::positionSensitive(DDLogicalPart& glog, DDCompactView& cpv) {
  int ncol = tileN_[0] / 2;
  int nrow = tileN_[1] / 2;
#ifdef EDM_ML_DEBUG
  int kount(0);
  edm::LogVerbatim("HGCalGeom") << glog.ddname() << " Row " << nrow << " Column " << ncol;
#endif
  for (int nr = -nrow; nr <= nrow; ++nr) {
    int inr = (nr >= 0) ? nr : -nr;
    double ypos = (nr >= 0) ? (inr - 0.5) * tileStep_[1] : -(inr - 0.5) * tileStep_[1];
    for (int nc = -ncol; nc <= ncol; ++nc) {
      int inc = (nc >= 0) ? nc : -nc;
      double xpos = (nc >= 0) ? (inc - 0.5) * tileStep_[0] : -(inc - 0.5) * tileStep_[0];
      if (nr != 0 && nc != 0) {
        DDTranslation tran(xpos, ypos, 0.0);
        DDRotation rotation;
        int copy = inr * AHCalParameters::kColumn_ + inc;
        if (nc < 0)
          copy += AHCalParameters::kRowColumn_;
        if (nr < 0)
          copy += AHCalParameters::kSignRowColumn_;
        DDName name = DDName(DDSplit(tile_).first, DDSplit(tile_).second);
        cpv.position(name, glog.ddname(), copy, tran, rotation);
#ifdef EDM_ML_DEBUG
        kount++;
        edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << name << " number " << copy << " positioned in "
                                      << glog.ddname() << " at " << tran << " with " << rotation;
#endif
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDAHcalModuleAlgo: " << kount << " tiles for " << glog.ddname();
#endif
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDAHcalModuleAlgo, "hgcal:DDAHcalModuleAlgo");
