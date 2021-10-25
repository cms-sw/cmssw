///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalMixLayer.cc
// Description: Geometry factory class for HGCal (Mix)
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
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalProperty.h"
#include "Geometry/HGCalCommonData/interface/HGCalTileIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"

#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

//#define EDM_ML_DEBUG
using namespace angle_units::operators;

class DDHGCalMixLayer : public DDAlgorithm {
public:
  DDHGCalMixLayer();

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

protected:
  void constructLayers(const DDLogicalPart&, DDCompactView& cpv);
  void positionMix(const DDLogicalPart& glog,
                   const std::string& nameM,
                   int copyM,
                   double thick,
                   const DDMaterial& matter,
                   DDCompactView& cpv);

private:
  HGCalGeomTools geomTools_;

  static constexpr double tol1_ = 0.01;
  static constexpr double tol2_ = 0.00001;

  int waferTypes_;                        // Number of wafer types
  int facingTypes_;                       // Types of facings of modules toward IP
  int partialTypes_;                      // Number of partial wafer types
  int orientationTypes_;                  // Number of partial wafer orienations
  int phiBinsScint_;                      // Maximum number of cells along phi
  int firstLayer_;                        // Copy # of the first sensitive layer
  int absorbMode_;                        // Absorber mode
  int sensitiveMode_;                     // Sensitive mode
  double zMinBlock_;                      // Starting z-value of the block
  double waferSize_;                      // Width of the wafer
  double waferSepar_;                     // Sensor separation
  int sectors_;                           // Sectors
  std::vector<double> slopeB_;            // Slope at the lower R
  std::vector<double> zFrontB_;           // Starting Z values for the slopes
  std::vector<double> rMinFront_;         // Corresponding rMin's
  std::vector<double> slopeT_;            // Slopes at the larger R
  std::vector<double> zFrontT_;           // Starting Z values for the slopes
  std::vector<double> rMaxFront_;         // Corresponding rMax's
  std::vector<std::string> waferFull_;    // Names of full wafer modules
  std::vector<std::string> waferPart_;    // Names of partial wafer modules
  std::vector<std::string> materials_;    // Materials
  std::vector<std::string> names_;        // Names
  std::vector<double> thick_;             // Thickness of the material
  std::vector<int> copyNumber_;           // Initial copy numbers
  std::vector<int> layers_;               // Number of layers in a section
  std::vector<double> layerThick_;        // Thickness of each section
  std::vector<int> layerType_;            // Type of the layer
  std::vector<int> layerSense_;           // Content of a layer (sensitive?)
  std::vector<std::string> materialTop_;  // Materials of top layers
  std::vector<std::string> namesTop_;     // Names of top layers
  std::vector<double> layerThickTop_;     // Thickness of the top sections
  std::vector<int> layerTypeTop_;         // Type of the Top layer
  std::vector<int> copyNumberTop_;        // Initial copy numbers (top section)
  std::vector<int> layerTypes_;           // Layer type of silicon layers
  std::vector<int> waferIndex_;           // Wafer index for the types
  std::vector<int> waferProperty_;        // Wafer property
  std::vector<int> waferLayerStart_;      // Start index of wafers in each layer
  std::vector<double> tileRMin_;          // Minimum radius of each ring
  std::vector<double> tileRMax_;          // Maximum radius of each ring
  std::vector<int> tileIndex_;            // Index of tile (layer/start|end ring)
  std::vector<int> tilePhis_;             // Tile phi range for each index
  std::vector<int> tileLayerStart_;       // Start index of tiles in each layer
  std::string nameSpace_;                 // Namespace of this and ALL sub-parts
  std::unordered_set<int> copies_;        // List of copy #'s
  double alpha_, cosAlpha_;
};

DDHGCalMixLayer::DDHGCalMixLayer() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: Creating an instance";
#endif
}

void DDHGCalMixLayer::initialize(const DDNumericArguments& nArgs,
                                 const DDVectorArguments& vArgs,
                                 const DDMapArguments&,
                                 const DDStringArguments& sArgs,
                                 const DDStringVectorArguments& vsArgs) {
  waferTypes_ = static_cast<int>(nArgs["WaferTypes"]);
  facingTypes_ = static_cast<int>(nArgs["FacingTypes"]);
  partialTypes_ = static_cast<int>(nArgs["PartialTypes"]);
  orientationTypes_ = static_cast<int>(nArgs["OrientationTypes"]);
  phiBinsScint_ = static_cast<int>(nArgs["NPhiBinScint"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer::Number of types of wafers: " << waferTypes_
                                << " facings: " << facingTypes_ << " partials: " << partialTypes_
                                << " Orientations: " << orientationTypes_ << "; number of cells along phi "
                                << phiBinsScint_;
#endif
  firstLayer_ = (int)(nArgs["FirstLayer"]);
  absorbMode_ = (int)(nArgs["AbsorberMode"]);
  sensitiveMode_ = (int)(nArgs["SensitiveMode"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer::First Layer " << firstLayer_ << " and "
                                << "Absober:Sensitive mode " << absorbMode_ << ":" << sensitiveMode_;
#endif
  zMinBlock_ = nArgs["zMinBlock"];
  waferSize_ = nArgs["waferSize"];
  waferSepar_ = nArgs["SensorSeparation"];
  sectors_ = (int)(nArgs["Sectors"]);
  alpha_ = (1._pi) / sectors_;
  cosAlpha_ = cos(alpha_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: zStart " << zMinBlock_ << " wafer width " << waferSize_
                                << " separations " << waferSepar_ << " sectors " << sectors_ << ":"
                                << convertRadToDeg(alpha_) << ":" << cosAlpha_;
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
  waferFull_ = vsArgs["WaferNamesFull"];
  waferPart_ = vsArgs["WaferNamesPartial"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: " << waferFull_.size() << " full and " << waferPart_.size()
                                << " partial modules\nDDHGCalMixLayer:Full Modules:";
  unsigned int i1max = static_cast<unsigned int>(waferFull_.size());
  for (unsigned int i1 = 0; i1 < i1max; i1 += 2) {
    std::ostringstream st1;
    unsigned int i2 = std::min((i1 + 2), i1max);
    for (unsigned int i = i1; i < i2; ++i)
      st1 << " [" << i << "] " << waferFull_[i];
    edm::LogVerbatim("HGCalGeom") << st1.str() << std::endl;
  }
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: Partial Modules:";
  i1max = static_cast<unsigned int>(waferPart_.size());
  for (unsigned int i1 = 0; i1 < i1max; i1 += 2) {
    std::ostringstream st1;
    unsigned int i2 = std::min((i1 + 2), i1max);
    for (unsigned int i = i1; i < i2; ++i)
      st1 << " [" << i << "] " << waferPart_[i];
    edm::LogVerbatim("HGCalGeom") << st1.str() << std::endl;
  }
#endif
  materials_ = vsArgs["MaterialNames"];
  names_ = vsArgs["VolumeNames"];
  thick_ = vArgs["Thickness"];
  copyNumber_.resize(materials_.size(), 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: " << materials_.size() << " types of volumes";
  for (unsigned int i = 0; i < names_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << names_[i] << " of thickness " << thick_[i]
                                  << " filled with " << materials_[i] << " first copy number " << copyNumber_[i];
#endif
  layers_ = dbl_to_int(vArgs["Layers"]);
  layerThick_ = vArgs["LayerThick"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "There are " << layers_.size() << " blocks";
  for (unsigned int i = 0; i < layers_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << layerThick_[i] << " with " << layers_[i]
                                  << " layers";
#endif
  layerType_ = dbl_to_int(vArgs["LayerType"]);
  layerSense_ = dbl_to_int(vArgs["LayerSense"]);
  layerTypes_ = dbl_to_int(vArgs["LayerTypes"]);
#ifdef EDM_ML_DEBUG
  for (unsigned int i = 0; i < layerTypes_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "LayerTypes [" << i << "] " << layerTypes_[i];
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
  materialTop_ = vsArgs["TopMaterialNames"];
  namesTop_ = vsArgs["TopVolumeNames"];
  layerThickTop_ = vArgs["TopLayerThickness"];
  layerTypeTop_ = dbl_to_int(vArgs["TopLayerType"]);
  copyNumberTop_.resize(materialTop_.size(), firstLayer_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: " << materialTop_.size() << " types of volumes in the top part";
  for (unsigned int i = 0; i < materialTop_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << namesTop_[i] << " of thickness " << layerThickTop_[i]
                                  << " filled with " << materialTop_[i] << " first copy number " << copyNumberTop_[i];
  edm::LogVerbatim("HGCalGeom") << "There are " << layerTypeTop_.size() << " layers in the top part";
  for (unsigned int i = 0; i < layerTypeTop_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerTypeTop_[i];
#endif
  waferIndex_ = dbl_to_int(vArgs["WaferIndex"]);
  waferProperty_ = dbl_to_int(vArgs["WaferProperties"]);
  waferLayerStart_ = dbl_to_int(vArgs["WaferLayerStart"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "waferProperties with " << waferIndex_.size() << " entries in "
                                << waferLayerStart_.size() << " layers";
  for (unsigned int k = 0; k < waferLayerStart_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "LayerStart[" << k << "] " << waferLayerStart_[k];
  for (unsigned int k = 0; k < waferIndex_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << waferIndex_[k] << " ("
                                  << HGCalWaferIndex::waferLayer(waferIndex_[k]) << ", "
                                  << HGCalWaferIndex::waferU(waferIndex_[k]) << ", "
                                  << HGCalWaferIndex::waferV(waferIndex_[k]) << ") : ("
                                  << HGCalProperty::waferThick(waferProperty_[k]) << ":"
                                  << HGCalProperty::waferPartial(waferProperty_[k]) << ":"
                                  << HGCalProperty::waferOrient(waferProperty_[k]) << ")";
#endif
  tileRMin_ = vArgs["TileRMin"];
  tileRMax_ = vArgs["TileRMax"];
  tileIndex_ = dbl_to_int(vArgs["TileLayerRings"]);
  tilePhis_ = dbl_to_int(vArgs["TilePhiRange"]);
  tileLayerStart_ = dbl_to_int(vArgs["TileLayerStart"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer:: with " << tileRMin_.size() << " rings";
  for (unsigned int k = 0; k < tileRMin_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "Ring[" << k << "] " << tileRMin_[k] << " : " << tileRMax_[k];
  edm::LogVerbatim("HGCalGeom") << "TileProperties with " << tileIndex_.size() << " entries in "
                                << tileLayerStart_.size() << " layers";
  for (unsigned int k = 0; k < tileLayerStart_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "LayerStart[" << k << "] " << tileLayerStart_[k];
  for (unsigned int k = 0; k < tileIndex_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << tileIndex_[k] << " ("
                                  << "Layer " << std::get<0>(HGCalTileIndex::tileUnpack(tileIndex_[k])) << " Ring "
                                  << std::get<1>(HGCalTileIndex::tileUnpack(tileIndex_[k])) << ":"
                                  << std::get<2>(HGCalTileIndex::tileUnpack(tileIndex_[k])) << ") Phi "
                                  << std::get<1>(HGCalTileIndex::tileUnpack(tilePhis_[k])) << ":"
                                  << std::get<2>(HGCalTileIndex::tileUnpack(tilePhis_[k]));
#endif
  nameSpace_ = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: NameSpace " << nameSpace_ << ":";
#endif
}

////////////////////////////////////////////////////////////////////
// DDHGCalMixLayer methods...
////////////////////////////////////////////////////////////////////

void DDHGCalMixLayer::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalMixLayer...";
  copies_.clear();
#endif
  constructLayers(parent(), cpv);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: " << copies_.size() << " different wafer copy numbers";
  int k(0);
  for (std::unordered_set<int>::const_iterator itr = copies_.begin(); itr != copies_.end(); ++itr, ++k) {
    edm::LogVerbatim("HGCalGeom") << "Copy [" << k << "] : " << (*itr);
  }
  copies_.clear();
  edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalMixLayer construction...";
#endif
}

void DDHGCalMixLayer::constructLayers(const DDLogicalPart& module, DDCompactView& cpv) {
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
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: Layer " << ly << ":" << ii << " Front " << zi << ", " << routF
                                    << " Back " << zo << ", " << rinB << " superlayer thickness " << layerThick_[i];
#endif
      DDName matName(DDSplit(materials_[ii]).first, DDSplit(materials_[ii]).second);
      DDMaterial matter(matName);
      DDLogicalPart glog;
      if (layerSense_[ly] < 1) {
        std::vector<double> pgonZ, pgonRin, pgonRout;
        double rmax =
            (std::min(routF, HGCalGeomTools::radius(zz + hthick, zFrontT_, rMaxFront_, slopeT_)) * cosAlpha_) - tol1_;
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
          if (layerSense_[ly] == 0 || absorbMode_ == 0)
            pgonRout[isec] = rmax;
          else
            pgonRout[isec] = pgonRout[isec] * cosAlpha_ - tol1_;
        }
        DDSolid solid =
            DDSolidFactory::polyhedra(DDName(name, nameSpace_), sectors_, -alpha_, 2._pi, pgonZ, pgonRin, pgonRout);
        glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: " << solid.name() << " polyhedra of " << sectors_
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
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: " << solid.name() << " Tubs made of " << matName
                                      << " of dimensions " << rinB << ":" << rins << ", " << routF << ":" << routs
                                      << ", " << hthick << ", 0.0, 360.0 and positioned in: " << glog.name()
                                      << " number " << copy;
#endif
        positionMix(glog, name, copy, thick_[ii], matter, cpv);
      }
      DDTranslation r1(0, 0, zz);
      DDRotation rot;
      cpv.position(glog, module, copy, r1, rot);
      ++copyNumber_[ii];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: " << glog.name() << " number " << copy << " positioned in "
                                    << module.name() << " at " << r1 << " with no rotation";
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

void DDHGCalMixLayer::positionMix(const DDLogicalPart& glog,
                                  const std::string& nameM,
                                  int copyM,
                                  double thick,
                                  const DDMaterial& matter,
                                  DDCompactView& cpv) {
  DDRotation rot;

  // Make the top part first
  for (unsigned int ly = 0; ly < layerTypeTop_.size(); ++ly) {
    int ii = layerTypeTop_[ly];
    copyNumberTop_[ii] = copyM;
  }
  double hthick = 0.5 * thick;
  double dphi = (2._pi) / phiBinsScint_;
  double thickTot(0), zpos(-hthick);
  for (unsigned int ly = 0; ly < layerTypeTop_.size(); ++ly) {
    int ii = layerTypeTop_[ly];
    int copy = copyNumberTop_[ii];
    int layer = copy - firstLayer_;
    double hthickl = 0.5 * layerThickTop_[ii];
    thickTot += layerThickTop_[ii];
    zpos += hthickl;
    DDName matName(DDSplit(materialTop_[ii]).first, DDSplit(materialTop_[ii]).second);
    DDMaterial matter1(matName);
    unsigned int k = 0;
    int firstTile = tileLayerStart_[layer];
    int lastTile = ((layer + 1 < static_cast<int>(tileLayerStart_.size())) ? tileLayerStart_[layer + 1]
                                                                           : static_cast<int>(tileIndex_.size()));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: Layer " << ly << ":" << ii << " Copy " << copy << " Tiles "
                                  << firstTile << ":" << lastTile;
#endif
    for (int ti = firstTile; ti < lastTile; ++ti) {
      double r1 = tileRMin_[std::get<1>(HGCalTileIndex::tileUnpack(tileIndex_[ti])) - 1];
      double r2 = tileRMax_[std::get<2>(HGCalTileIndex::tileUnpack(tileIndex_[ti])) - 1];
      int fimin = std::get<1>(HGCalTileIndex::tileUnpack(tilePhis_[ti]));
      int fimax = std::get<2>(HGCalTileIndex::tileUnpack(tilePhis_[ti]));
      double phi1 = dphi * (fimin - 1);
      double phi2 = dphi * (fimax - fimin + 1);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: Layer " << copy << " iR "
                                    << std::get<1>(HGCalTileIndex::tileUnpack(tileIndex_[ly])) << ":"
                                    << std::get<2>(HGCalTileIndex::tileUnpack(tileIndex_[ly])) << " R " << r1 << ":"
                                    << r2 << " Thick " << (2.0 * hthickl) << " phi " << fimin << ":" << fimax << ":"
                                    << convertRadToDeg(phi1) << ":" << convertRadToDeg(phi2);
      ;
#endif
      std::string name = namesTop_[ii] + "L" + std::to_string(copy) + "F" + std::to_string(k);
      ++k;
      DDSolid solid = DDSolidFactory::tubs(DDName(name, nameSpace_), hthickl, r1, r2, phi1, phi2);
      DDLogicalPart glog1 = DDLogicalPart(solid.ddname(), matter1, solid);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: " << glog1.name() << " Tubs made of " << matName
                                    << " of dimensions " << r1 << ", " << r2 << ", " << hthickl << ", "
                                    << convertRadToDeg(phi1) << ", " << convertRadToDeg(phi2);
#endif
      DDTranslation tran(0, 0, zpos);
      cpv.position(glog1, glog, copy, tran, rot);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: Position " << glog1.name() << " number " << copy << " in "
                                    << glog.name() << " at " << tran << " with no rotation";
#endif
    }
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
  int layer = (copyM - firstLayer_);
  static const double sqrt3 = std::sqrt(3.0);
  int layercenter = (layerTypes_[layer] == HGCalTypes::CornerCenteredLambda)
                        ? 1
                        : ((layerTypes_[layer] == HGCalTypes::CornerCenteredY) ? 2 : 0);
  int layerType = (layerTypes_[layer] == HGCalTypes::WaferCenteredBack) ? 1 : 0;
  int firstWafer = waferLayerStart_[layer];
  int lastWafer = ((layer + 1 < static_cast<int>(waferLayerStart_.size())) ? waferLayerStart_[layer + 1]
                                                                           : static_cast<int>(waferIndex_.size()));
  double r = 0.5 * (waferSize_ + waferSepar_);
  double R = 2.0 * r / sqrt3;
  double dy = 0.75 * R;
  const auto& xyoff = geomTools_.shiftXY(layercenter, (waferSize_ + waferSepar_));
#ifdef EDM_ML_DEBUG
  int ium(0), ivm(0), kount(0);
  std::vector<int> ntype(3, 0);
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: " << glog.ddname() << "  r " << r << " R " << R << " dy " << dy
                                << " Shift " << xyoff.first << ":" << xyoff.second << " WaferSize "
                                << (waferSize_ + waferSepar_) << " index " << firstWafer << ":" << (lastWafer - 1);
#endif
  for (int k = firstWafer; k < lastWafer; ++k) {
    int u = HGCalWaferIndex::waferU(waferIndex_[k]);
    int v = HGCalWaferIndex::waferV(waferIndex_[k]);
#ifdef EDM_ML_DEBUG
    int iu = std::abs(u);
    int iv = std::abs(v);
#endif
    int nr = 2 * v;
    int nc = -2 * u + v;
    double xpos = xyoff.first + nc * r;
    double ypos = xyoff.second + nr * dy;
    int type = HGCalProperty::waferThick(waferProperty_[k]);
    int part = HGCalProperty::waferPartial(waferProperty_[k]);
    int orien = HGCalProperty::waferOrient(waferProperty_[k]);
    std::string wafer;
    int i(999);
    if (part == HGCalTypes::WaferFull) {
      i = layerType * waferTypes_ + type;
      wafer = waferFull_[i];
    } else {
      i = (part - 1) * waferTypes_ * facingTypes_ * orientationTypes_ + layerType * waferTypes_ * orientationTypes_ +
          type * orientationTypes_ + orien;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << " layertype:type:part:orien:ind " << layerType << ":" << type << ":" << part
                                    << ":" << orien << ":" << i << ":" << waferPart_.size();
#endif
      wafer = waferPart_[i];
    }
    int copy = HGCalTypes::packTypeUV(type, u, v);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << " DDHGCalMixLayer: Layer " << HGCalWaferIndex::waferLayer(waferIndex_[k])
                                  << " Wafer " << wafer << " number " << copy << " type :part:orien:ind " << type << ":"
                                  << part << ":" << orien << ":" << i << " layer:u:v " << (layer + firstLayer_) << ":"
                                  << u << ":" << v;
    if (iu > ium)
      ium = iu;
    if (iv > ivm)
      ivm = iv;
    kount++;
    if (copies_.count(copy) == 0)
      copies_.insert(copy);
#endif
    DDTranslation tran(xpos, ypos, 0.0);
    DDName name = DDName(DDSplit(wafer).first, DDSplit(wafer).second);
    cpv.position(name, glog.ddname(), copy, tran, rot);
#ifdef EDM_ML_DEBUG
    ++ntype[type];
    edm::LogVerbatim("HGCalGeom") << " DDHGCalMixLayer: " << name << " number " << copy << " type " << layerType << ":"
                                  << type << " positioned in " << glog.ddname() << " at " << tran
                                  << " with no rotation";
#endif
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixLayer: Maximum # of u " << ium << " # of v " << ivm << " and " << kount
                                << " wafers (" << ntype[0] << ":" << ntype[1] << ":" << ntype[2] << ") for "
                                << glog.ddname();
#endif
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalMixLayer, "hgcal:DDHGCalMixLayer");
