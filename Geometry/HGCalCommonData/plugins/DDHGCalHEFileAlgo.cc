///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalHEFileAlgo.cc
// Description: Geometry factory class for HGCal (Mix)
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/Math/interface/CMSUnits.h"
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
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalProperty.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"

#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

//#define EDM_ML_DEBUG
using namespace cms_units::operators;

class DDHGCalHEFileAlgo : public DDAlgorithm {
public:
  DDHGCalHEFileAlgo();

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

protected:
  void constructLayers(const DDLogicalPart&, DDCompactView& cpv);
  void positionMix(const DDLogicalPart& glog,
                   const std::string& name,
                   int copy,
                   double thick,
                   const DDMaterial& matter,
                   double rin,
                   double rmid,
                   double routF,
                   double zz,
                   DDCompactView& cpv);
  void positionSensitive(
      const DDLogicalPart& glog, double rin, double rout, double zpos, int layertype, int layer, DDCompactView& cpv);

private:
  HGCalGeomTools geomTools_;

  static constexpr double tol1_ = 0.01;
  static constexpr double tol2_ = 0.00001;

  std::vector<std::string> wafers_;        // Wafers
  std::vector<std::string> materials_;     // Materials
  std::vector<std::string> names_;         // Names
  std::vector<double> thick_;              // Thickness of the material
  std::vector<int> copyNumber_;            // Initial copy numbers
  std::vector<int> layers_;                // Number of layers in a section
  std::vector<double> layerThick_;         // Thickness of each section
  std::vector<double> rMixLayer_;          // Partition between Si/Sci part
  std::vector<int> layerType_;             // Type of the layer
  std::vector<int> layerSense_;            // Content of a layer (sensitive?)
  int firstLayer_;                         // Copy # of the first sensitive layer
  int absorbMode_;                         // Absorber mode
  int sensitiveMode_;                      // Sensitive mode
  std::vector<std::string> materialsTop_;  // Materials of top layers
  std::vector<std::string> namesTop_;      // Names of top layers
  std::vector<double> layerThickTop_;      // Thickness of the top sections
  std::vector<int> layerTypeTop_;          // Type of the Top layer
  std::vector<int> copyNumberTop_;         // Initial copy numbers (top section)
  std::vector<std::string> materialsBot_;  // Materials of bottom layers
  std::vector<std::string> namesBot_;      // Names of bottom layers
  std::vector<double> layerThickBot_;      // Thickness of the bottom sections
  std::vector<int> layerTypeBot_;          // Type of the bottom layers
  std::vector<int> copyNumberBot_;         // Initial copy numbers (bot section)
  std::vector<int> layerSenseBot_;         // Content of bottom layer (sensitive?)
  std::vector<int> layerCenter_;           // Centering of the wafers

  double zMinBlock_;                // Starting z-value of the block
  std::vector<int> waferIndex_;     // Wafer index for the types
  std::vector<int> waferProperty_;  // Wafer property
  double waferSize_;                // Width of the wafer
  double waferSepar_;               // Sensor separation
  int sectors_;                     // Sectors
  std::vector<double> slopeB_;      // Slope at the lower R
  std::vector<double> zFrontB_;     // Starting Z values for the slopes
  std::vector<double> rMinFront_;   // Corresponding rMin's
  std::vector<double> slopeT_;      // Slopes at the larger R
  std::vector<double> zFrontT_;     // Starting Z values for the slopes
  std::vector<double> rMaxFront_;   // Corresponding rMax's
  std::string nameSpace_;           // Namespace of this and ALL sub-parts
  std::unordered_set<int> copies_;  // List of copy #'s
  double alpha_, cosAlpha_;
};

DDHGCalHEFileAlgo::DDHGCalHEFileAlgo() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: Creating an instance";
#endif
}

void DDHGCalHEFileAlgo::initialize(const DDNumericArguments& nArgs,
                                   const DDVectorArguments& vArgs,
                                   const DDMapArguments&,
                                   const DDStringArguments& sArgs,
                                   const DDStringVectorArguments& vsArgs) {
  wafers_ = vsArgs["WaferNames"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: " << wafers_.size() << " wafers";
  for (unsigned int i = 0; i < wafers_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << "] " << wafers_[i];
#endif
  materials_ = vsArgs["MaterialNames"];
  names_ = vsArgs["VolumeNames"];
  thick_ = vArgs["Thickness"];
  copyNumber_.resize(materials_.size(), 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: " << materials_.size() << " types of volumes";
  for (unsigned int i = 0; i < names_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << names_[i] << " of thickness " << thick_[i]
                                  << " filled with " << materials_[i] << " first copy number " << copyNumber_[i];
#endif
  layers_ = dbl_to_int(vArgs["Layers"]);
  layerThick_ = vArgs["LayerThick"];
  rMixLayer_ = vArgs["LayerRmix"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "There are " << layers_.size() << " blocks";
  for (unsigned int i = 0; i < layers_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << layerThick_[i] << " Rmid " << rMixLayer_[i]
                                  << " with " << layers_[i] << " layers";
#endif
  layerType_ = dbl_to_int(vArgs["LayerType"]);
  layerSense_ = dbl_to_int(vArgs["LayerSense"]);
  firstLayer_ = (int)(nArgs["FirstLayer"]);
  absorbMode_ = (int)(nArgs["AbsorberMode"]);
  sensitiveMode_ = (int)(nArgs["SensitiveMode"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "First Layer " << firstLayer_ << " and "
                                << "Absober:Sensitive mode " << absorbMode_ << ":" << sensitiveMode_;
#endif
  layerCenter_ = dbl_to_int(vArgs["LayerCenter"]);
#ifdef EDM_ML_DEBUG
  for (unsigned int i = 0; i < layerCenter_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "LayerCenter [" << i << "] " << layerCenter_[i];
#endif
  if (firstLayer_ > 0) {
    for (unsigned int i = 0; i < layerType_.size(); ++i) {
      if (layerSense_[i] > 0) {
        int ii = layerType_[i];
        copyNumber_[ii] = firstLayer_;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "First copy number for layer type " << i << ":" << ii << " with "
                                      << materials_[ii] << " changed to " << copyNumber_[ii];
#endif
        break;
      }
    }
  } else {
    firstLayer_ = 1;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "There are " << layerType_.size() << " layers";
  for (unsigned int i = 0; i < layerType_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerType_[i] << " sensitive class "
                                  << layerSense_[i];
#endif
  materialsTop_ = vsArgs["TopMaterialNames"];
  namesTop_ = vsArgs["TopVolumeNames"];
  layerThickTop_ = vArgs["TopLayerThickness"];
  layerTypeTop_ = dbl_to_int(vArgs["TopLayerType"]);
  copyNumberTop_.resize(materialsTop_.size(), 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: " << materialsTop_.size() << " types of volumes in the top part";
  for (unsigned int i = 0; i < materialsTop_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << namesTop_[i] << " of thickness " << layerThickTop_[i]
                                  << " filled with " << materialsTop_[i] << " first copy number " << copyNumberTop_[i];
  edm::LogVerbatim("HGCalGeom") << "There are " << layerTypeTop_.size() << " layers in the top part";
  for (unsigned int i = 0; i < layerTypeTop_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerTypeTop_[i];
#endif
  materialsBot_ = vsArgs["BottomMaterialNames"];
  namesBot_ = vsArgs["BottomVolumeNames"];
  layerTypeBot_ = dbl_to_int(vArgs["BottomLayerType"]);
  layerSenseBot_ = dbl_to_int(vArgs["BottomLayerSense"]);
  layerThickBot_ = vArgs["BottomLayerThickness"];
  copyNumberBot_.resize(materialsBot_.size(), 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: " << materialsBot_.size()
                                << " types of volumes in the bottom part";
  for (unsigned int i = 0; i < materialsBot_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << namesBot_[i] << " of thickness " << layerThickBot_[i]
                                  << " filled with " << materialsBot_[i] << " first copy number " << copyNumberBot_[i];
  edm::LogVerbatim("HGCalGeom") << "There are " << layerTypeBot_.size() << " layers in the bottom part";
  for (unsigned int i = 0; i < layerTypeBot_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerTypeBot_[i]
                                  << " sensitive class " << layerSenseBot_[i];
#endif
  zMinBlock_ = nArgs["zMinBlock"];
  waferSize_ = nArgs["waferSize"];
  waferSepar_ = nArgs["SensorSeparation"];
  sectors_ = (int)(nArgs["Sectors"]);
  alpha_ = (1._pi) / sectors_;
  cosAlpha_ = cos(alpha_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: zStart " << zMinBlock_ << " wafer width " << waferSize_
                                << " separations " << waferSepar_ << " sectors " << sectors_ << ":"
                                << convertRadToDeg(alpha_) << ":" << cosAlpha_;
#endif
  waferIndex_ = dbl_to_int(vArgs["WaferIndex"]);
  waferProperty_ = dbl_to_int(vArgs["WaferProperties"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "waferProperties with " << waferIndex_.size() << " entries";
  for (unsigned int k = 0; k < waferIndex_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << waferIndex_[k] << " ("
                                  << HGCalWaferIndex::waferLayer(waferIndex_[k]) << ", "
                                  << HGCalWaferIndex::waferU(waferIndex_[k]) << ", "
                                  << HGCalWaferIndex::waferV(waferIndex_[k]) << ") : ("
                                  << HGCalProperty::waferThick(waferProperty_[k]) << ":"
                                  << HGCalProperty::waferPartial(waferProperty_[k]) << ":"
                                  << HGCalProperty::waferOrient(waferProperty_[k]) << ")";
#endif
  slopeB_ = vArgs["SlopeBottom"];
  zFrontB_ = vArgs["ZFrontBottom"];
  rMinFront_ = vArgs["RMinFront"];
  slopeT_ = vArgs["SlopeTop"];
  zFrontT_ = vArgs["ZFrontTop"];
  rMaxFront_ = vArgs["RMaxFront"];
#ifdef EDM_ML_DEBUG
  for (unsigned int i = 0; i < slopeB_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] Zmin " << zFrontB_[i] << " Rmin " << rMinFront_[i]
                                  << " Slope " << slopeB_[i];
  for (unsigned int i = 0; i < slopeT_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] Zmin " << zFrontT_[i] << " Rmax " << rMaxFront_[i]
                                  << " Slope " << slopeT_[i];
#endif
  nameSpace_ = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: NameSpace " << nameSpace_;
#endif
}

////////////////////////////////////////////////////////////////////
// DDHGCalHEFileAlgo methods...
////////////////////////////////////////////////////////////////////

void DDHGCalHEFileAlgo::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalHEFileAlgo...";
  copies_.clear();
#endif
  constructLayers(parent(), cpv);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: " << copies_.size() << " different wafer copy numbers";
  int k(0);
  for (std::unordered_set<int>::const_iterator itr = copies_.begin(); itr != copies_.end(); ++itr, ++k) {
    edm::LogVerbatim("HGCalGeom") << "Copy [" << k << "] : " << (*itr);
  }
  copies_.clear();
  edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalHEFileAlgo construction...";
#endif
}

void DDHGCalHEFileAlgo::constructLayers(const DDLogicalPart& module, DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: \t\tInside Layers";
#endif
  double zi(zMinBlock_);
  int laymin(0);
  for (unsigned int i = 0; i < layers_.size(); i++) {
    double zo = zi + layerThick_[i];
    double routF = HGCalGeomTools::radius(zi, zFrontT_, rMaxFront_, slopeT_);
    int laymax = laymin + layers_[i];
    double zz = zi;
    double thickTot(0);
    for (int ly = laymin; ly < laymax; ++ly) {
      int ii = layerType_[ly];
      int copy = copyNumber_[ii];
      double hthick = 0.5 * thick_[ii];
      double rinB = HGCalGeomTools::radius(zo, zFrontB_, rMinFront_, slopeB_);
      zz += hthick;
      thickTot += thick_[ii];

      std::string name = names_[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: Layer " << ly << ":" << ii << " Front " << zi << ", "
                                    << routF << " Back " << zo << ", " << rinB << " superlayer thickness "
                                    << layerThick_[i];
#endif
      DDName matName(DDSplit(materials_[ii]).first, DDSplit(materials_[ii]).second);
      DDMaterial matter(matName);
      DDLogicalPart glog;
      if (layerSense_[ly] < 1) {
        std::vector<double> pgonZ, pgonRin, pgonRout;
        if (layerSense_[ly] == 0 || absorbMode_ == 0) {
          double rmax =
              (std::min(routF, HGCalGeomTools::radius(zz + hthick, zFrontT_, rMaxFront_, slopeT_)) * cosAlpha_) - tol1_;
          pgonZ.emplace_back(-hthick);
          pgonZ.emplace_back(hthick);
          pgonRin.emplace_back(rinB);
          pgonRin.emplace_back(rinB);
          pgonRout.emplace_back(rmax);
          pgonRout.emplace_back(rmax);
        } else {
          HGCalGeomTools::radius(zz - hthick,
                                 zz + hthick,
                                 zFrontB_,
                                 rMinFront_,
                                 slopeB_,
                                 zFrontT_,
                                 rMaxFront_,
                                 slopeT_,
                                 -layerSense_[ly],
                                 pgonZ,
                                 pgonRin,
                                 pgonRout);
          for (unsigned int isec = 0; isec < pgonZ.size(); ++isec) {
            pgonZ[isec] -= zz;
            pgonRout[isec] = pgonRout[isec] * cosAlpha_ - tol1_;
          }
        }
        DDSolid solid =
            DDSolidFactory::polyhedra(DDName(name, nameSpace_), sectors_, -alpha_, 2._pi, pgonZ, pgonRin, pgonRout);
        glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: " << solid.name() << " polyhedra of " << sectors_
                                      << " sectors covering " << convertRadToDeg(-alpha_) << ":"
                                      << convertRadToDeg(-alpha_ + 2._pi) << " with " << pgonZ.size() << " sections";
        for (unsigned int k = 0; k < pgonZ.size(); ++k)
          edm::LogVerbatim("HGCalGeom") << "[" << k << "] z " << pgonZ[k] << " R " << pgonRin[k] << ":" << pgonRout[k];
#endif
      } else {
        double rins = (sensitiveMode_ < 1) ? rinB : HGCalGeomTools::radius(zz + hthick, zFrontB_, rMinFront_, slopeB_);
        double routs =
            (sensitiveMode_ < 1) ? routF : HGCalGeomTools::radius(zz - hthick, zFrontT_, rMaxFront_, slopeT_);
        DDSolid solid = DDSolidFactory::tubs(DDName(name, nameSpace_), hthick, rins, routs, 0.0, 2._pi);
        glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: " << solid.name() << " Tubs made of " << matName
                                      << " of dimensions " << rinB << ":" << rins << ", " << routF << ":" << routs
                                      << ", " << hthick << ", 0.0, 360.0 and positioned in: " << glog.name()
                                      << " number " << copy;
#endif
        positionMix(glog, name, copy, thick_[ii], matter, rins, rMixLayer_[i], routs, zz, cpv);
      }
      DDTranslation r1(0, 0, zz);
      DDRotation rot;
      cpv.position(glog, module, copy, r1, rot);
      ++copyNumber_[ii];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: " << glog.name() << " number " << copy << " positioned in "
                                    << module.name() << " at " << r1 << " with " << rot;
#endif
      zz += hthick;
    }  // End of loop over layers in a block
    zi = zo;
    laymin = laymax;
    // Make consistency check of all the partitions of the block
    if (std::abs(thickTot - layerThick_[i]) >= tol2_) {
      if (thickTot > layerThick_[i]) {
        edm::LogError("HGCalGeom") << "Thickness of the partition " << layerThick_[i] << " is smaller than " << thickTot
                                   << ": thickness of all its components **** ERROR ****";
      } else {
        edm::LogWarning("HGCalGeom") << "Thickness of the partition " << layerThick_[i] << " does not match with "
                                     << thickTot << " of the components";
      }
    }
  }  // End of loop over blocks
}

void DDHGCalHEFileAlgo::positionMix(const DDLogicalPart& glog,
                                    const std::string& nameM,
                                    int copyM,
                                    double thick,
                                    const DDMaterial& matter,
                                    double rin,
                                    double rmid,
                                    double rout,
                                    double zz,
                                    DDCompactView& cpv) {
  DDLogicalPart glog1;
  DDTranslation tran;
  DDRotation rot;
  for (unsigned int ly = 0; ly < layerTypeTop_.size(); ++ly) {
    int ii = layerTypeTop_[ly];
    copyNumberTop_[ii] = copyM;
  }
  for (unsigned int ly = 0; ly < layerTypeBot_.size(); ++ly) {
    int ii = layerTypeBot_[ly];
    copyNumberBot_[ii] = copyM;
  }
  double hthick = 0.5 * thick;
  // Make the top part first
  std::string name = nameM + "Top";
  DDSolid solid = DDSolidFactory::tubs(DDName(name, nameSpace_), hthick, rmid, rout, 0.0, 2._pi);
  glog1 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: " << solid.name() << " Tubs made of " << matter.name()
                                << " of dimensions " << rmid << ", " << rout << ", " << hthick << ", 0.0, 360.0";
#endif
  cpv.position(glog1, glog, 1, tran, rot);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: " << glog1.name() << " number 1 positioned in " << glog.name()
                                << " at " << tran << " with " << rot;
#endif
  double thickTot(0), zpos(-hthick);
  for (unsigned int ly = 0; ly < layerTypeTop_.size(); ++ly) {
    int ii = layerTypeTop_[ly];
    int copy = copyNumberTop_[ii];
    double hthickl = 0.5 * layerThickTop_[ii];
    thickTot += layerThickTop_[ii];
    name = namesTop_[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: Layer " << ly << ":" << ii << " R " << rmid << ":" << rout
                                  << " Thick " << layerThickTop_[ii];
#endif
    DDName matName(DDSplit(materialsTop_[ii]).first, DDSplit(materialsTop_[ii]).second);
    DDMaterial matter1(matName);
    solid = DDSolidFactory::tubs(DDName(name, nameSpace_), hthickl, rmid, rout, 0.0, 2._pi);
    DDLogicalPart glog2 = DDLogicalPart(solid.ddname(), matter1, solid);
#ifdef EDM_ML_DEBUG
    double eta1 = -log(tan(0.5 * atan(rmid / zz)));
    double eta2 = -log(tan(0.5 * atan(rout / zz)));
    edm::LogVerbatim("HGCalGeom") << name << " z|rin|rout " << zz << ":" << rmid << ":" << rout << " eta " << eta1
                                  << ":" << eta2;
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: " << solid.name() << " Tubs made of " << matName
                                  << " of dimensions " << rmid << ", " << rout << ", " << hthickl << ", 0.0, 360.0";
#endif
    zpos += hthickl;
    DDTranslation r1(0, 0, zpos);
    cpv.position(glog2, glog1, copy, r1, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: Position " << glog2.name() << " number " << copy << " in "
                                  << glog1.name() << " at " << r1 << " with " << rot;
#endif
    ++copyNumberTop_[ii];
    zpos += hthickl;
  }
  if (std::abs(thickTot - thick) >= tol2_) {
    if (thickTot > thick) {
      edm::LogError("HGCalGeom") << "Thickness of the partition " << thick << " is smaller than " << thickTot
                                 << ": thickness of all its components in the top part **** ERROR ****";
    } else {
      edm::LogWarning("HGCalGeom") << "Thickness of the partition " << thick << " does not match with " << thickTot
                                   << " of the components in top part";
    }
  }

  // Make the bottom part next
  name = nameM + "Bottom";
  solid = DDSolidFactory::tubs(DDName(name, nameSpace_), hthick, rin, rmid, 0.0, 2._pi);
  glog1 = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: " << solid.name() << " Tubs made of " << matter.name()
                                << " of dimensions " << rin << ", " << rmid << ", " << hthick << ", 0.0, 360.0";
#endif
  cpv.position(glog1, glog, 1, tran, rot);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: " << glog1.name() << " number 1 positioned in " << glog.name()
                                << " at " << tran << " with " << rot;
#endif
  thickTot = 0;
  zpos = -hthick;
  for (unsigned int ly = 0; ly < layerTypeBot_.size(); ++ly) {
    int ii = layerTypeBot_[ly];
    int copy = copyNumberBot_[ii];
    double hthickl = 0.5 * layerThickBot_[ii];
    thickTot += layerThickBot_[ii];
    name = namesBot_[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: Layer " << ly << ":" << ii << " R " << rin << ":" << rmid
                                  << " Thick " << layerThickBot_[ii];
#endif
    DDName matName(DDSplit(materialsBot_[ii]).first, DDSplit(materialsBot_[ii]).second);
    DDMaterial matter1(matName);
    solid = DDSolidFactory::tubs(DDName(name, nameSpace_), hthickl, rin, rmid, 0.0, 2._pi);
    DDLogicalPart glog2 = DDLogicalPart(solid.ddname(), matter1, solid);
#ifdef EDM_ML_DEBUG
    double eta1 = -log(tan(0.5 * atan(rin / zz)));
    double eta2 = -log(tan(0.5 * atan(rmid / zz)));
    edm::LogVerbatim("HGCalGeom") << name << " z|rin|rout " << zz << ":" << rin << ":" << rmid << " eta " << eta1 << ":"
                                  << eta2;
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: " << solid.name() << " Tubs made of " << matName
                                  << " of dimensions " << rin << ", " << rmid << ", " << hthickl << ", 0.0, 360.0";
#endif
    zpos += hthickl;
    DDTranslation r1(0, 0, zpos);
    cpv.position(glog2, glog1, copy, r1, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: Position " << glog2.name() << " number " << copy << " in "
                                  << glog1.name() << " at " << r1 << " with " << rot;
#endif
    if (layerSenseBot_[ly] != 0) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: z " << (zz + zpos) << " Center " << copy << ":"
                                    << (copy - firstLayer_) << ":" << layerCenter_[copy - firstLayer_];
#endif
      positionSensitive(glog2, rin, rmid, zz + zpos, layerSenseBot_[ly], (copy - firstLayer_), cpv);
    }
    zpos += hthickl;
    ++copyNumberBot_[ii];
  }
  if (std::abs(thickTot - thick) >= tol2_) {
    if (thickTot > thick) {
      edm::LogError("HGCalGeom") << "Thickness of the partition " << thick << " is smaller than " << thickTot
                                 << ": thickness of all its components in the top part **** ERROR ****";
    } else {
      edm::LogWarning("HGCalGeom") << "Thickness of the partition " << thick << " does not match with " << thickTot
                                   << " of the components in top part";
    }
  }
}

void DDHGCalHEFileAlgo::positionSensitive(
    const DDLogicalPart& glog, double rin, double rout, double zpos, int layertype, int layer, DDCompactView& cpv) {
  static const double sqrt3 = std::sqrt(3.0);
  int layercenter = layerCenter_[layer];
  double r = 0.5 * (waferSize_ + waferSepar_);
  double R = 2.0 * r / sqrt3;
  double dy = 0.75 * R;
  int N = (int)(0.5 * rout / r) + 2;
  const auto& xyoff = geomTools_.shiftXY(layercenter, (waferSize_ + waferSepar_));
#ifdef EDM_ML_DEBUG
  int ium(0), ivm(0), iumAll(0), ivmAll(0), kount(0), ntot(0), nin(0);
  std::vector<int> ntype(6, 0);
  edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: " << glog.ddname() << " rin:rout " << rin << ":" << rout
                                << " zpos " << zpos << " N " << N << " for maximum u, v Offset; Shift " << xyoff.first
                                << ":" << xyoff.second << " WaferSize " << (waferSize_ + waferSepar_);
#endif
  for (int u = -N; u <= N; ++u) {
    for (int v = -N; v <= N; ++v) {
#ifdef EDM_ML_DEBUG
      int iu = std::abs(u);
      int iv = std::abs(v);
#endif
      int nr = 2 * v;
      int nc = -2 * u + v;
      double xpos = xyoff.first + nc * r;
      double ypos = xyoff.second + nr * dy;
      const auto& corner = HGCalGeomTools::waferCorner(xpos, ypos, r, R, rin, rout, false);
#ifdef EDM_ML_DEBUG
      ++ntot;
#endif
      int indx = HGCalWaferIndex::waferIndex((layer + firstLayer_), u, v, false);
      int type = HGCalWaferType::getType(indx, waferIndex_, waferProperty_);
      if (corner.first > 0 && type >= 0) {
        int copy = HGCalTypes::packTypeUV(type, u, v);
        if (layertype > 1)
          type += 3;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << " DDHGCalHEFileAlgo: " << wafers_[type] << " number " << copy << " type "
                                      << type << " layer:u:v:indx " << (layer + firstLayer_) << ":" << u << ":" << v
                                      << ":" << indx;
        if (iu > ium)
          ium = iu;
        if (iv > ivm)
          ivm = iv;
        kount++;
        if (copies_.count(copy) == 0)
          copies_.insert(copy);
#endif
        if (corner.first == (int)(HGCalParameters::k_CornerSize)) {
#ifdef EDM_ML_DEBUG
          if (iu > iumAll)
            iumAll = iu;
          if (iv > ivmAll)
            ivmAll = iv;
          ++nin;
#endif
          DDTranslation tran(xpos, ypos, 0.0);
          DDRotation rotation;
          DDName name = DDName(DDSplit(wafers_[type]).first, DDSplit(wafers_[type]).second);
          cpv.position(name, glog.ddname(), copy, tran, rotation);
#ifdef EDM_ML_DEBUG
          ++ntype[type];
          edm::LogVerbatim("HGCalGeom") << " DDHGCalHEFileAlgo: " << name << " number " << copy << " type " << layertype
                                        << ":" << type << " positioned in " << glog.ddname() << " at " << tran
                                        << " with " << rotation;
#endif
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalHEFileAlgo: Maximum # of u " << ium << ":" << iumAll << " # of v " << ivm
                                << ":" << ivmAll << " and " << nin << ":" << kount << ":" << ntot << " wafers ("
                                << ntype[0] << ":" << ntype[1] << ":" << ntype[2] << ":" << ntype[3] << ":" << ntype[4]
                                << ":" << ntype[5] << ") for " << glog.ddname() << " R " << rin << ":" << rout;
#endif
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalHEFileAlgo, "hgcal:DDHGCalHEFileAlgo");
