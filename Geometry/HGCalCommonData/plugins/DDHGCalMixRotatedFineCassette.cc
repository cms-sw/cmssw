///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalMixRotatedFineFineCassette.cc
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
#include "Geometry/HGCalCommonData/interface/HGCalCell.h"
#include "Geometry/HGCalCommonData/interface/HGCalCassette.h"
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

class DDHGCalMixRotatedFineCassette : public DDAlgorithm {
public:
  DDHGCalMixRotatedFineCassette();

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
                   int absType,
                   bool fine,
                   DDCompactView& cpv);
  void testCassetteShift();

private:
  HGCalGeomTools geomTools_;
  HGCalCassette cassette_;

  static constexpr double tol0_ = 0.0001;
  static constexpr double tol1_ = 0.01;
  static constexpr double tol2_ = 0.00001;

  int waferTypes_;                         // Number of wafer types
  int passiveTypes_;                       // Number of passive types
  int facingTypes_;                        // Types of facings of modules toward IP
  int orientationTypes_;                   // Number of partial wafer orienations
  int partialTypes_;                       // Number of partial types
  int placeOffset_;                        // Offset for placement
  int phiBinsScint_;                       // Maximum number of cells along phi (coarse)
  int phiBinsFineScint_;                   // Maximum number of cells along phi (fine)
  int firstFineLayer_;                     // Copy # of the first Fine sensitive layer
  int firstCoarseLayer_;                   // Copy # of the first Coarse sensitive layer
  int absorbMode_;                         // Absorber mode
  int sensitiveMode_;                      // Sensitive mode
  int passiveMode_;                        // Mode for passive components
  double zMinBlock_;                       // Starting z-value of the block
  double waferSize_;                       // Width of the wafer
  double waferSepar_;                      // Sensor separation
  int sectors_;                            // Sectors
  int cassettes_;                          // Cassettes
  std::vector<double> slopeB_;             // Slope at the lower R
  std::vector<double> zFrontB_;            // Starting Z values for the slopes
  std::vector<double> rMinFront_;          // Corresponding rMin's
  std::vector<double> slopeT_;             // Slopes at the larger R
  std::vector<double> zFrontT_;            // Starting Z values for the slopes
  std::vector<double> rMaxFront_;          // Corresponding rMax's
  std::vector<std::string> waferFull_;     // Names of full wafer modules
  std::vector<std::string> waferPart_;     // Names of partial wafer modules
  std::vector<std::string> passiveFull_;   // Names of full passive modules
  std::vector<std::string> passivePart_;   // Names of partial passive modules
  std::vector<std::string> materials_;     // Materials
  std::vector<std::string> names_;         // Names
  std::vector<double> thick_;              // Thickness of the material
  std::vector<int> copyNumber_;            // Initial copy numbers
  std::vector<int> layers_;                // Number of layers in a section
  std::vector<double> layerThick_;         // Thickness of each section
  std::vector<int> layerType_;             // Type of the layer
  std::vector<int> layerSense_;            // Content of a layer (sensitive?)
  std::vector<std::string> materialTop_;   // Materials of top layers
  std::vector<std::string> namesTop_;      // Names of top layers
  std::vector<double> layerThickTop_;      // Thickness of the top sections
  std::vector<int> layerTypeTop_;          // Type of the Top layer
  std::vector<int> copyNumberTop_;         // Initial copy numbers (top section)
  int coverTypeTop_;                       // Type of the Top layer cover
  int coverTopLayers_;                     // Number of cover layers in top section
  std::vector<int> copyNumberCoverTop_;    // Initial copy number of top cover
  std::vector<int> layerOrient_;           // Layer orientation for the silicon component
  std::vector<int> waferIndex_;            // Wafer index for the types
  std::vector<int> waferProperty_;         // Wafer property
  std::vector<int> waferLayerStart_;       // Start index of wafers in each layer
  std::vector<double> cassetteShift_;      // Shifts of the cassetes
  std::vector<double> tileFineRMin_;       // Minimum radius of each fine ring
  std::vector<double> tileFineRMax_;       // Maximum radius of each fine ring
  std::vector<int> tileFineIndex_;         // Index of tile (layer/start|end fine ring)
  std::vector<int> tileFinePhis_;          // Tile phi range for each index in fine ring
  std::vector<int> tileFineLayerStart_;    // Start index of tiles in each fine layer
  std::vector<double> tileCoarseRMin_;     // Minimum radius of each coarse ring
  std::vector<double> tileCoarseRMax_;     // Maximum radius of each coarse ring
  std::vector<int> tileCoarseIndex_;       // Index of tile (layer/start|end coarse ring)
  std::vector<int> tileCoarsePhis_;        // Tile phi range for each index in coarse ring
  std::vector<int> tileCoarseLayerStart_;  // Start index of tiles in each coarse layer
  std::vector<double> retract_;            // Radial retraction of he tiles
  std::vector<double> cassetteShiftScnt_;  // Shifts of the cassetes for scintillators
  std::string nameSpace_;                  // Namespace of this and ALL sub-parts
  std::unordered_set<int> copies_;         // List of copy #'s
  double alpha_, cosAlpha_;
};

DDHGCalMixRotatedFineCassette::DDHGCalMixRotatedFineCassette() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Creating an instance";
#endif
}

void DDHGCalMixRotatedFineCassette::initialize(const DDNumericArguments& nArgs,
                                               const DDVectorArguments& vArgs,
                                               const DDMapArguments&,
                                               const DDStringArguments& sArgs,
                                               const DDStringVectorArguments& vsArgs) {
  waferTypes_ = static_cast<int>(nArgs["WaferTypes"]);
  passiveTypes_ = static_cast<int>(nArgs["PassiveTypes"]);
  facingTypes_ = static_cast<int>(nArgs["FacingTypes"]);
  orientationTypes_ = static_cast<int>(nArgs["OrientationTypes"]);
  partialTypes_ = static_cast<int>(nArgs["PartialTypes"]);
  placeOffset_ = static_cast<int>(nArgs["PlaceOffset"]);
  phiBinsScint_ = static_cast<int>(nArgs["NPhiBinScint"]);
  phiBinsFineScint_ = static_cast<int>(nArgs["NPhiBinFineScint"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette::Number of types of wafers: " << waferTypes_
                                << " passives: " << passiveTypes_ << " facings: " << facingTypes_
                                << " Orientations: " << orientationTypes_ << " PartialTypes: " << partialTypes_
                                << " PlaceOffset: " << placeOffset_ << "; number of cells along phi "
                                << phiBinsFineScint_ << ":" << phiBinsScint_;
#endif
  firstFineLayer_ = static_cast<int>(nArgs["FirstFineLayer"]);
  firstCoarseLayer_ = static_cast<int>(nArgs["FirstCoarseLayer"]);
  absorbMode_ = static_cast<int>(nArgs["AbsorberMode"]);
  sensitiveMode_ = static_cast<int>(nArgs["SensitiveMode"]);
  passiveMode_ = static_cast<int>(nArgs["PassiveMode"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette::First Layers " << firstFineLayer_ << ":"
                                << firstCoarseLayer_ << " and Absober:Sensitive mode " << absorbMode_ << ":"
                                << sensitiveMode_ << ":" << passiveMode_;
#endif
  zMinBlock_ = nArgs["zMinBlock"];
  waferSize_ = nArgs["waferSize"];
  waferSepar_ = nArgs["SensorSeparation"];
  sectors_ = static_cast<int>(nArgs["Sectors"]);
  cassettes_ = static_cast<int>(nArgs["Cassettes"]);
  alpha_ = (1._pi) / sectors_;
  cosAlpha_ = cos(alpha_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: zStart " << zMinBlock_ << " wafer width "
                                << waferSize_ << " separations " << waferSepar_ << " sectors " << sectors_ << ":"
                                << convertRadToDeg(alpha_) << ":" << cosAlpha_ << " with " << cassettes_
                                << " cassettes";
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
  waferFull_ = vsArgs["WaferNamesFull"];
  waferPart_ = vsArgs["WaferNamesPartial"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << waferFull_.size() << " full and "
                                << waferPart_.size() << " partial modules\nDDHGCalMixRotatedFineCassette:Full Modules:";
  unsigned int i1max = static_cast<unsigned int>(waferFull_.size());
  for (unsigned int i1 = 0; i1 < i1max; i1 += 2) {
    std::ostringstream st1;
    unsigned int i2 = std::min((i1 + 2), i1max);
    for (unsigned int i = i1; i < i2; ++i)
      st1 << " [" << i << "] " << waferFull_[i];
    edm::LogVerbatim("HGCalGeom") << st1.str();
  }
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Partial Modules:";
  i1max = static_cast<unsigned int>(waferPart_.size());
  for (unsigned int i1 = 0; i1 < i1max; i1 += 2) {
    std::ostringstream st1;
    unsigned int i2 = std::min((i1 + 2), i1max);
    for (unsigned int i = i1; i < i2; ++i)
      st1 << " [" << i << "] " << waferPart_[i];
    edm::LogVerbatim("HGCalGeom") << st1.str();
  }
#endif
  passiveFull_ = vsArgs["PassiveNamesFull"];
  passivePart_ = vsArgs["PassiveNamesPartial"];
  if (passiveFull_.size() <= 1)
    passiveFull_.clear();
  if (passivePart_.size() <= 1)
    passivePart_.clear();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: " << passiveFull_.size() << " full and "
                                << passivePart_.size() << " partial passive modules";
  i1max = static_cast<unsigned int>(passiveFull_.size());
  for (unsigned int i1 = 0; i1 < i1max; i1 += 2) {
    std::ostringstream st1;
    unsigned int i2 = std::min((i1 + 2), i1max);
    for (unsigned int i = i1; i < i2; ++i)
      st1 << " [" << i << "] " << passiveFull_[i];
    edm::LogVerbatim("HGCalGeom") << st1.str();
  }
  edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: Partial Modules:";
  i1max = static_cast<unsigned int>(passivePart_.size());
  for (unsigned int i1 = 0; i1 < i1max; i1 += 2) {
    std::ostringstream st1;
    unsigned int i2 = std::min((i1 + 2), i1max);
    for (unsigned int i = i1; i < i2; ++i)
      st1 << " [" << i << "] " << passivePart_[i];
    edm::LogVerbatim("HGCalGeom") << st1.str();
  }
#endif
  materials_ = vsArgs["MaterialNames"];
  names_ = vsArgs["VolumeNames"];
  thick_ = vArgs["Thickness"];
  copyNumber_.resize(materials_.size(), 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << materials_.size() << " types of volumes";
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
  layerOrient_ = dbl_to_int(vArgs["LayerTypes"]);
  for (unsigned int k = 0; k < layerOrient_.size(); ++k)
    layerOrient_[k] = HGCalTypes::layerType(layerOrient_[k]);
#ifdef EDM_ML_DEBUG
  for (unsigned int i = 0; i < layerOrient_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "LayerOrient [" << i << "] " << layerOrient_[i];
#endif
  if (firstFineLayer_ > 0) {
    for (unsigned int i = 0; i < layerType_.size(); ++i) {
      if (layerSense_[i] != 0) {
        int ii = layerType_[i];
        copyNumber_[ii] = firstFineLayer_;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "First copy number for layer type " << i << ":" << ii << " with "
                                      << materials_[ii] << " changed to " << copyNumber_[ii];
#endif
      }
    }
  } else {
    firstFineLayer_ = 1;
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
  copyNumberTop_.resize(materialTop_.size(), firstFineLayer_);
  coverTypeTop_ = static_cast<int>(nArgs["TopCoverLayerType"]);
  coverTopLayers_ = static_cast<int>(nArgs["TopCoverLayers"]);
  copyNumberCoverTop_.resize(coverTopLayers_, firstFineLayer_);
#ifdef EDM_ML_DEBUG
  std::ostringstream st0;
  for (int k = 0; k < coverTopLayers_; ++k)
    st0 << " " << copyNumberCoverTop_[k];
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << materialTop_.size()
                                << " types of volumes in the top part; " << coverTopLayers_ << " covers of Type "
                                << coverTypeTop_ << " with initial copy numbers: " << st0.str();
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
  cassetteShift_ = vArgs["CassetteShift"];
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
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << cassetteShift_.size()
                                << " elements for cassette shifts";
  unsigned int j1max = cassetteShift_.size();
  for (unsigned int j1 = 0; j1 < j1max; j1 += 6) {
    std::ostringstream st1;
    unsigned int j2 = std::min((j1 + 6), j1max);
    for (unsigned int j = j1; j < j2; ++j)
      st1 << " [" << j << "] " << std::setw(9) << cassetteShift_[j];
    edm::LogVerbatim("HGCalGeom") << st1.str();
  }
#endif
  tileFineRMin_ = vArgs["Tile6RMin"];
  tileFineRMax_ = vArgs["Tile6RMax"];
  tileFineIndex_ = dbl_to_int(vArgs["Tile6LayerRings"]);
  tileFinePhis_ = dbl_to_int(vArgs["Tile6PhiRange"]);
  tileFineLayerStart_ = dbl_to_int(vArgs["Tile6LayerStart"]);
  tileCoarseRMin_ = vArgs["TileRMin"];
  tileCoarseRMax_ = vArgs["TileRMax"];
  tileCoarseIndex_ = dbl_to_int(vArgs["TileLayerRings"]);
  tileCoarsePhis_ = dbl_to_int(vArgs["TilePhiRange"]);
  tileCoarseLayerStart_ = dbl_to_int(vArgs["TileLayerStart"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette:: with " << tileFineRMin_.size() << ":"
                                << tileCoarseRMin_.size() << " rings";
  for (unsigned int k = 0; k < tileFineRMin_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "Fine Ring[" << k << "] " << tileFineRMin_[k] << " : " << tileFineRMax_[k];
  for (unsigned int k = 0; k < tileCoarseRMin_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "Coarse Ring[" << k << "] " << tileCoarseRMin_[k] << " : " << tileCoarseRMax_[k];
  edm::LogVerbatim("HGCalGeom") << "TileProperties with " << tileFineIndex_.size() << ":" << tileCoarseIndex_.size()
                                << " entries in " << tileFineLayerStart_.size() << ":" << tileCoarseLayerStart_.size()
                                << " layers";
  for (unsigned int k = 0; k < tileFineLayerStart_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "FineLayerStart[" << k << "] " << tileFineLayerStart_[k];
  for (unsigned int k = 0; k < tileCoarseLayerStart_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "CoarseLayerStart[" << k << "] " << tileCoarseLayerStart_[k];
  for (unsigned int k = 0; k < tileFineIndex_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << tileFineIndex_[k] << " ("
                                  << "Layer " << std::get<0>(HGCalTileIndex::tileUnpack(tileFineIndex_[k])) << " Ring "
                                  << std::get<1>(HGCalTileIndex::tileUnpack(tileFineIndex_[k])) << ":"
                                  << std::get<2>(HGCalTileIndex::tileUnpack(tileFineIndex_[k])) << ") Phi "
                                  << std::get<1>(HGCalTileIndex::tileUnpack(tileFinePhis_[k])) << ":"
                                  << std::get<2>(HGCalTileIndex::tileUnpack(tileFinePhis_[k]));
  for (unsigned int k = 0; k < tileCoarseIndex_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << tileCoarseIndex_[k] << " ("
                                  << "Layer " << std::get<0>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[k]))
                                  << " Ring " << std::get<1>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[k])) << ":"
                                  << std::get<2>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[k])) << ") Phi "
                                  << std::get<1>(HGCalTileIndex::tileUnpack(tileCoarsePhis_[k])) << ":"
                                  << std::get<2>(HGCalTileIndex::tileUnpack(tileCoarsePhis_[k]));
#endif
  retract_ = vArgs["ScintRetract"];
  double dphi = M_PI / cassettes_;
  for (unsigned int k = 0; k < layers_.size(); ++k) {
    for (int j = 0; j < cassettes_; ++j) {
      double phi = (2 * j + 1) * dphi;
      cassetteShiftScnt_.emplace_back(retract_[k] * cos(phi));
      cassetteShiftScnt_.emplace_back(retract_[k] * sin(phi));
    }
  }
#ifdef EDM_ML_DEBUG
  unsigned int j2max = cassetteShiftScnt_.size();
  for (unsigned int j1 = 0; j1 < j2max; j1 += 6) {
    std::ostringstream st1;
    unsigned int j2 = std::min((j1 + 6), j2max);
    for (unsigned int j = j1; j < j2; ++j)
      st1 << " [" << j << "] " << std::setw(9) << cassetteShiftScnt_[j];
    edm::LogVerbatim("HGCalGeom") << "Scintillator Cassette shiftt " << convertRadToDeg(dphi) << " " << st1.str();
  }
#endif
  nameSpace_ = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: NameSpace " << nameSpace_ << ":";
#endif
  cassette_.setParameter(cassettes_, cassetteShift_, false);
  cassette_.setParameterScint(cassetteShiftScnt_);
  cassette_.setParameterRetract(retract_);

  int testCassette = static_cast<int>(nArgs["TestCassetteShift"]);
  if (testCassette != 0)
    testCassetteShift();
}

////////////////////////////////////////////////////////////////////
// DDHGCalMixRotatedFineCassette methods...
////////////////////////////////////////////////////////////////////

void DDHGCalMixRotatedFineCassette::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalMixRotatedFineCassette...";
  copies_.clear();
#endif
  constructLayers(parent(), cpv);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << copies_.size()
                                << " different wafer copy numbers";
  int k(0);
  for (std::unordered_set<int>::const_iterator itr = copies_.begin(); itr != copies_.end(); ++itr, ++k) {
    edm::LogVerbatim("HGCalGeom") << "Copy [" << k << "] : " << (*itr);
  }
  copies_.clear();
  edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalMixRotatedFineCassette construction...";
#endif
}

void DDHGCalMixRotatedFineCassette::constructLayers(const DDLogicalPart& module, DDCompactView& cpv) {
  double zi(zMinBlock_);
  int laymin(0);
  unsigned int fineLayers = static_cast<unsigned int>(firstCoarseLayer_ - firstFineLayer_);
  for (unsigned int i = 0; i < layers_.size(); i++) {
    double zo = zi + layerThick_[i];
    double routF = HGCalGeomTools::radius(zi, zFrontT_, rMaxFront_, slopeT_);
    int laymax = laymin + layers_[i];
    double zz = zi;
    double thickTot(0);
    bool fine = (i < fineLayers) ? true : false;
    for (int ly = laymin; ly < laymax; ++ly) {
      int ii = layerType_[ly];
      int copy = copyNumber_[ii];
      double hthick = 0.5 * thick_[ii];
      double rinB = HGCalGeomTools::radius(zo, zFrontB_, rMinFront_, slopeB_);
      zz += hthick;
      thickTot += thick_[ii];

      std::string name = names_[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Layer " << ly << ":" << ii << " Front " << zi
                                    << ", " << routF << " Back " << zo << ", " << rinB << " superlayer thickness "
                                    << layerThick_[i];
#endif
      DDName matName(DDSplit(materials_[ii]).first, DDSplit(materials_[ii]).second);
      DDMaterial matter(matName);
      DDLogicalPart glog;
      if (layerSense_[ly] <= 0) {
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
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << solid.name() << " polyhedra of "
                                      << sectors_ << " sectors covering " << convertRadToDeg(-alpha_) << ":"
                                      << convertRadToDeg(-alpha_ + 2._pi) << " with " << pgonZ.size() << " sections";
        for (unsigned int k = 0; k < pgonZ.size(); ++k)
          edm::LogVerbatim("HGCalGeom") << "[" << k << "] z " << pgonZ[k] << " R " << pgonRin[k] << ":" << pgonRout[k];
        edm::LogVerbatim("HGCalGeom") << "LayeerSense " << layerSense_[ly];
#endif
        if (layerSense_[ly] < 0) {
          int absType = -layerSense_[ly];
          unsigned int num = (absType <= waferTypes_) ? passiveFull_.size() : passivePart_.size();
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "Abstype " << absType << " num " << num;
#endif
          if (num > 1)
            positionMix(glog, name, copy, thick_[ii], matter, absType, fine, cpv);
        }
      } else {
        double rins = (sensitiveMode_ < 1) ? rinB : HGCalGeomTools::radius(zz + hthick, zFrontB_, rMinFront_, slopeB_);
        double routs =
            (sensitiveMode_ < 1) ? routF : HGCalGeomTools::radius(zz - hthick, zFrontT_, rMaxFront_, slopeT_);
        DDSolid solid = DDSolidFactory::tubs(DDName(name, nameSpace_), hthick, rins, routs, 0.0, 2._pi);
        glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << solid.name() << " Tubs made of "
                                      << matName << " of dimensions " << rinB << ":" << rins << ", " << routF << ":"
                                      << routs << ", " << hthick << ", 0.0, 360.0 and positioned in: " << glog.name()
                                      << " number " << copy;
#endif
        positionMix(glog, name, copy, thick_[ii], matter, -layerSense_[ly], fine, cpv);
      }
      DDTranslation r1(0, 0, zz);
      DDRotation rot;
      cpv.position(glog, module, copy, r1, rot);
      ++copyNumber_[ii];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << glog.name() << " number " << copy
                                    << " positioned in " << module.name() << " at " << r1 << " with no rotation";
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

void DDHGCalMixRotatedFineCassette::positionMix(const DDLogicalPart& glog,
                                                const std::string& nameM,
                                                int copyM,
                                                double thick,
                                                const DDMaterial& matter,
                                                int absType,
                                                bool fine,
                                                DDCompactView& cpv) {
  DDRotation rot;

  // Make the top part first
  for (unsigned int ly = 0; ly < layerTypeTop_.size(); ++ly) {
    int ii = layerTypeTop_[ly];
    copyNumberTop_[ii] = copyM;
  }
  double hthick = 0.5 * thick;
  double dphi = (fine) ? ((2._pi) / phiBinsFineScint_) : ((2._pi) / phiBinsScint_);
  double thickTot(0), zpos(-hthick);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Entry to positionMix with Name " << nameM << " copy "
                                << copyM << " Thick " << thick << " AbsType " << absType << " Fine " << fine << " dphi "
                                << convertRadToDeg(dphi);
#endif
  if (absType < 0) {
    for (unsigned int ly = 0; ly < layerTypeTop_.size(); ++ly) {
      int ii = layerTypeTop_[ly];
      int copy = copyNumberTop_[ii];
      int layer = (fine) ? (copy - firstFineLayer_) : (copy - firstCoarseLayer_);
      int layer0 = (copy - std::min(firstFineLayer_, firstCoarseLayer_) + 1);
      double hthickl = 0.5 * layerThickTop_[ii];
      thickTot += layerThickTop_[ii];
      zpos += hthickl;
      DDName matName(DDSplit(materialTop_[ii]).first, DDSplit(materialTop_[ii]).second);
      DDMaterial matter1(matName);
      unsigned int k = 0;
      int firstTile = (fine) ? tileFineLayerStart_[layer] : tileCoarseLayerStart_[layer];
      int lastTile = (fine) ? ((layer + 1 < static_cast<int>(tileFineLayerStart_.size()))
                                   ? tileFineLayerStart_[layer + 1]
                                   : static_cast<int>(tileFineIndex_.size()))
                            : ((layer + 1 < static_cast<int>(tileCoarseLayerStart_.size()))
                                   ? tileCoarseLayerStart_[layer + 1]
                                   : static_cast<int>(tileCoarseIndex_.size()));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Layer " << ly << ":" << ii << ":" << layer << ":"
                                    << layer0 << " Copy " << copy << " Tiles " << firstTile << ":" << lastTile
                                    << " Size " << tileFineIndex_.size() << ":" << tileCoarseIndex_.size() << " Fine "
                                    << fine << " absType " << absType;
      int cassette;
#endif
      for (int ti = firstTile; ti < lastTile; ++ti) {
        double r1, r2;
        int fimin, fimax;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: ti " << ti << ":" << fine << " index "
                                      << tileFineIndex_.size() << ":" << tileCoarseIndex_.size() << " Phis "
                                      << tileFinePhis_.size() << ":" << tileCoarsePhis_.size();
#endif
        if (fine) {
          r1 = tileFineRMin_[std::get<1>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti])) - 1];
          r2 = tileFineRMax_[std::get<2>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti])) - 1];
#ifdef EDM_ML_DEBUG
          cassette = std::get<0>(HGCalTileIndex::tileUnpack(tileFinePhis_[ti]));
#endif
          fimin = std::get<1>(HGCalTileIndex::tileUnpack(tileFinePhis_[ti]));
          fimax = std::get<2>(HGCalTileIndex::tileUnpack(tileFinePhis_[ti]));
        } else {
          r1 = tileCoarseRMin_[std::get<1>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti])) - 1];
          r2 = tileCoarseRMax_[std::get<2>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti])) - 1];
#ifdef EDM_ML_DEBUG
          cassette = std::get<0>(HGCalTileIndex::tileUnpack(tileCoarsePhis_[ti]));
#endif
          fimin = std::get<1>(HGCalTileIndex::tileUnpack(tileCoarsePhis_[ti]));
          fimax = std::get<2>(HGCalTileIndex::tileUnpack(tileCoarsePhis_[ti]));
        }
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Casstee|Fimin|Fimax " << cassette << ":"
                                      << fimin << ":" << fimax;
#endif
        double phi1 = dphi * (fimin - 1);
        double phi2 = dphi * (fimax - fimin + 1);
        r1 += retract_[layer0 - 1];
        r2 += retract_[layer0 - 1];
#ifdef EDM_ML_DEBUG
        double phi = phi1 + 0.5 * phi2;
        edm::LogVerbatim("HGCalGeom") << "1Layer " << ly << ":" << ii << ":" << copy << ":" << layer0 << " phi " << phi
                                      << " shift " << retract_[layer0 - 1];
        int cassette0 = HGCalCassette::cassetteType(2, 1, cassette);  //
        int ir1 = (fine) ? std::get<1>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti]))
                         : std::get<1>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti]));
        int ir2 = (fine) ? std::get<2>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti]))
                         : std::get<2>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti]));
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Layer " << copy << ":" << layer0 << " iR "
                                      << ir1 << ":" << ir2 << " R " << r1 << ":" << r2 << " Thick " << (2.0 * hthickl)
                                      << " phi " << fimin << ":" << fimax << ":" << convertRadToDeg(phi1) << ":"
                                      << convertRadToDeg(phi2) << " cassette " << cassette << ":" << cassette0
                                      << " Shift " << retract_[layer0 - 1];
#endif
        std::string name = namesTop_[ii] + "L" + std::to_string(copy) + "F" + std::to_string(k);
        ++k;
        DDSolid solid = DDSolidFactory::tubs(DDName(name, nameSpace_), hthickl, r1, r2, phi1, phi2);
        DDLogicalPart glog1 = DDLogicalPart(solid.ddname(), matter1, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << glog1.name() << " Tubs made of "
                                      << matName << " of dimensions " << r1 << ", " << r2 << ", " << hthickl << ", "
                                      << convertRadToDeg(phi1) << ", " << convertRadToDeg(phi2);
#endif
        DDTranslation tran(0, 0, zpos);
        cpv.position(glog1, glog, copy, tran, rot);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Position " << glog1.name() << " number "
                                      << copy << " in " << glog.name() << " at " << tran << " with no rotation";
#endif
      }
      ++copyNumberTop_[ii];
      zpos += hthickl;
    }
    if (std::abs(thickTot - thick) >= tol2_) {
      if (thickTot > thick) {
        edm::LogError("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Thickness of the partition " << thick
                                   << " is smaller than " << thickTot
                                   << ": thickness of all its components in the top part **** ERROR ****";
      } else {
        edm::LogWarning("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Thickness of the partition " << thick
                                     << " does not match with " << thickTot << " of the components in top part";
      }
    }
  } else {
    int ii = coverTypeTop_;
    int copy = copyNumberCoverTop_[absType - 1];
    int layer = (fine) ? (copy - firstFineLayer_) : (copy - firstCoarseLayer_);
    int layer0 = (copy - std::min(firstFineLayer_, firstCoarseLayer_) + 1);
    double hthickl = 0.5 * layerThickTop_[ii];
    zpos += hthickl;
    DDName matName(DDSplit(materialTop_[ii]).first, DDSplit(materialTop_[ii]).second);
    DDMaterial matter1(matName);
    unsigned int k = 0;
    int firstTile = (fine) ? tileFineLayerStart_[layer] : tileCoarseLayerStart_[layer];
    int lastTile =
        (fine) ? ((layer + 1 < static_cast<int>(tileFineLayerStart_.size())) ? tileFineLayerStart_[layer + 1]
                                                                             : static_cast<int>(tileFineIndex_.size()))
               : ((layer + 1 < static_cast<int>(tileCoarseLayerStart_.size()))
                      ? tileCoarseLayerStart_[layer + 1]
                      : static_cast<int>(tileCoarseIndex_.size()));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: TOP Layer " << ii << ":" << layer << ":" << layer0
                                  << " Copy " << copy << " Tiles " << firstTile << ":" << lastTile << " Size "
                                  << tileFineIndex_.size() << ":" << tileCoarseIndex_.size() << " Fine " << fine
                                  << " absType " << absType;
#endif
    for (int ti = firstTile; ti < lastTile; ++ti) {
      double r1, r2;
      int fimin, fimax;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: ti " << ti << ":" << fine << " index "
                                    << tileFineIndex_.size() << ":" << tileCoarseIndex_.size() << " Phis "
                                    << tileFinePhis_.size() << ":" << tileCoarsePhis_.size();
      int cassette;
#endif
      if (fine) {
        r1 = tileFineRMin_[std::get<1>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti])) - 1];
        r2 = tileFineRMax_[std::get<2>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti])) - 1];
#ifdef EDM_ML_DEBUG
        cassette = std::get<0>(HGCalTileIndex::tileUnpack(tileFinePhis_[ti]));
#endif
        fimin = std::get<1>(HGCalTileIndex::tileUnpack(tileFinePhis_[ti]));
        fimax = std::get<2>(HGCalTileIndex::tileUnpack(tileFinePhis_[ti]));
      } else {
        r1 = tileCoarseRMin_[std::get<1>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti])) - 1];
        r2 = tileCoarseRMax_[std::get<2>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti])) - 1];
#ifdef EDM_ML_DEBUG
        cassette = std::get<0>(HGCalTileIndex::tileUnpack(tileCoarsePhis_[ti]));
#endif
        fimin = std::get<1>(HGCalTileIndex::tileUnpack(tileCoarsePhis_[ti]));
        fimax = std::get<2>(HGCalTileIndex::tileUnpack(tileCoarsePhis_[ti]));
      }
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Casstee|Fimin|Fimax " << cassette << ":" << fimin
                                    << ":" << fimax;
#endif
      double phi1 = dphi * (fimin - 1);
      double phi2 = dphi * (fimax - fimin + 1);
      r1 += retract_[layer0 - 1];
      r2 += retract_[layer0 - 1];
#ifdef EDM_ML_DEBUG
      double phi = phi1 + 0.5 * phi2;
      edm::LogVerbatim("HGCalGeom") << "2Layer " << ii << ":" << copy << ":" << layer << ":" << layer0 << " phi " << phi
                                    << " shift " << retract_[layer0 - 1];
      int cassette0 = HGCalCassette::cassetteType(2, 1, cassette);  //
      int ir1 = (fine) ? std::get<1>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti]))
                       : std::get<1>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti]));
      int ir2 = (fine) ? std::get<2>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti]))
                       : std::get<2>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti]));
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Layer " << copy << ":" << (layer + 1) << ":"
                                    << layer0 << " iR " << ir1 << ":" << ir2 << " R " << r1 << ":" << r2 << " Thick "
                                    << (2.0 * hthickl) << " phi " << fimin << ":" << fimax << ":"
                                    << convertRadToDeg(phi1) << ":" << convertRadToDeg(phi2) << " cassette " << cassette
                                    << ":" << cassette0 << " Shift " << retract_[layer0 - 1];
#endif
      std::string name = namesTop_[ii] + "L" + std::to_string(copy) + "F" + std::to_string(k);
      ++k;
      DDSolid solid = DDSolidFactory::tubs(DDName(name, nameSpace_), hthickl, r1, r2, phi1, phi2);
      DDLogicalPart glog1 = DDLogicalPart(solid.ddname(), matter1, solid);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << glog1.name() << " Tubs made of " << matName
                                    << " of dimensions " << r1 << ", " << r2 << ", " << hthickl << ", "
                                    << convertRadToDeg(phi1) << ", " << convertRadToDeg(phi2);
#endif
      DDTranslation tran(0, 0, zpos);
      cpv.position(glog1, glog, copy, tran, rot);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Position " << glog1.name() << " number " << copy
                                    << " in " << glog.name() << " at " << tran << " with no rotation";
#endif
    }
    ++copyNumberCoverTop_[absType - 1];
  }

  // Make the bottom part next
  int layer = (copyM - firstFineLayer_);
  int layer0 = (copyM - std::min(firstFineLayer_, firstCoarseLayer_) + 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Start bottom section for layer " << (layer + 1)
                                << ":" << layer0 << " absType " << absType;
#endif
  if (absType > 0) {
#ifdef EDM_ML_DEBUG
    int kount(0);
#endif
    for (int k = 0; k < cassettes_; ++k) {
      int cassette = k + 1;
      auto cshift = cassette_.getShift(layer0, -1, cassette, false);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "3Layer " << layer << ":" << layer0 << " Cassette " << cassette << " shift "
                                    << cshift.first << ":" << cshift.second;
#endif
      double xpos = -cshift.first;
      double ypos = cshift.second;
      int i = layer * cassettes_ + k;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette::Passive: layer " << layer + 1 << " cassette "
                                    << cassette << " Shift " << cshift.first << ":" << cshift.second << " PassiveIndex "
                                    << i << ":" << passiveFull_.size() << ":" << passivePart_.size();
#endif
      std::string passive = (absType <= waferTypes_) ? passiveFull_[i] : passivePart_[i];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Passive " << passive << " number " << cassette
                                    << " pos " << xpos << ":" << ypos;
      kount++;
#endif
      DDTranslation tran(xpos, ypos, 0.0);
      DDRotation rotation;
      DDName name = DDName(DDSplit(passive).first, DDSplit(passive).second);
      cpv.position(name, glog.ddname(), cassette, tran, rotation);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << name << " number " << cassette
                                    << " positioned in " << glog.ddname() << " at " << tran << " with no rotation";
#endif
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << kount << " passives of type " << absType
                                  << " for " << glog.ddname();
#endif
  } else {
    static const double sqrt3 = std::sqrt(3.0);
    int layercenter = layerOrient_[layer];
    int layertype = HGCalTypes::layerFrontBack(layerOrient_[layer]);
    int firstWafer = waferLayerStart_[layer];
    int lastWafer = ((layer + 1 < static_cast<int>(waferLayerStart_.size())) ? waferLayerStart_[layer + 1]
                                                                             : static_cast<int>(waferIndex_.size()));
    double delx = 0.5 * (waferSize_ + waferSepar_);
    double dely = 2.0 * delx / sqrt3;
    double dy = 0.75 * dely;
    const auto& xyoff = geomTools_.shiftXY(layercenter, (waferSize_ + waferSepar_));
#ifdef EDM_ML_DEBUG
    int ium(0), ivm(0), kount(0);
    std::vector<int> ntype(3, 0);
    edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette::Bottom: " << glog.ddname() << "  r " << delx
                                  << " R " << dely << " dy " << dy << " Shift " << xyoff.first << ":" << xyoff.second
                                  << " WaferSize " << (waferSize_ + waferSepar_) << " index " << firstWafer << ":"

                                  << (lastWafer - 1) << " Copy " << copyM << ":" << layer;
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
      int type = HGCalProperty::waferThick(waferProperty_[k]);
      int part = HGCalProperty::waferPartial(waferProperty_[k]);
      int orien = HGCalProperty::waferOrient(waferProperty_[k]);
      int cassette = HGCalProperty::waferCassette(waferProperty_[k]);
      int place = HGCalCell::cellPlacementIndex(1, layertype, orien);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom")
          << "DDHGCalMixRotatedFineCassette::index:Property:layertype:type:part:orien:cassette:place:offsets:ind " << k
          << ":" << waferProperty_[k] << ":" << layertype << ":" << type << ":" << part << ":" << orien << ":"
          << cassette << ":" << place;
#endif
      auto cshift = cassette_.getShift(layer0, -1, cassette, false);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Layer " << (layer + 1) << ":" << layer0 << " Cassette " << cassette << " shift "
                                    << cshift.first << ":" << cshift.second;
#endif
      double xpos = xyoff.first - cshift.first + nc * delx;
      double ypos = xyoff.second + cshift.second + nr * dy;
#ifdef EDM_ML_DEBUG
      double xorig = xyoff.first + nc * delx;
      double yorig = xyoff.second + nr * dy;
      double angle = std::atan2(yorig, xorig);
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette::Wafer: layer " << layer + 1 << ":" << layer0
                                    << " cassette " << cassette << " Shift " << cshift.first << ":" << cshift.second
                                    << " Original " << xorig << ":" << yorig << ":" << convertRadToDeg(angle)
                                    << " Final " << xpos << ":" << ypos;
#endif
      std::string wafer;
      int i(999);
      if (part == HGCalTypes::WaferFull) {
        i = type * facingTypes_ * orientationTypes_ + place - placeOffset_;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << " FullWafer type:place:ind " << type << ":" << place << ":" << i << ":"
                                      << waferFull_.size();
#endif
        wafer = waferFull_[i];
      } else {
        int partoffset =
            (part >= HGCalTypes::WaferHDTop) ? HGCalTypes::WaferPartHDOffset : HGCalTypes::WaferPartLDOffset;
        i = (part - partoffset) * facingTypes_ * orientationTypes_ +
            HGCalTypes::WaferTypeOffset[type] * facingTypes_ * orientationTypes_ + place - placeOffset_;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << " layertype:type:part:orien:cassette:place:offsets:ind " << layertype << ":"
                                      << type << ":" << part << ":" << orien << ":" << cassette << ":" << place << ":"
                                      << partoffset << ":" << HGCalTypes::WaferTypeOffset[type] << ":" << i << ":"
                                      << waferPart_.size();
#endif
        wafer = waferPart_[i];
      }
      int copy = HGCalTypes::packTypeUV(type, u, v);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << " DDHGCalMixRotatedFineCassette: Layer "
                                    << HGCalWaferIndex::waferLayer(waferIndex_[k]) << " Wafer " << wafer << " number "
                                    << copy << " type :part:orien:ind " << type << ":" << part << ":" << orien << ":"
                                    << i << " layer:u:v " << (layer + firstFineLayer_) << ":" << u << ":" << v;
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
      edm::LogVerbatim("HGCalGeom") << " DDHGCalMixRotatedFineCassette: " << name << " number " << copy << " type "
                                    << layertype << ":" << type << " positioned in " << glog.ddname() << " at " << tran
                                    << " with no rotation";
#endif
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Maximum # of u " << ium << " # of v " << ivm
                                  << " and " << kount << " wafers (" << ntype[0] << ":" << ntype[1] << ":" << ntype[2]
                                  << ") for " << glog.ddname();
#endif
  }
}

void DDHGCalMixRotatedFineCassette::testCassetteShift() {
  for (unsigned int k = 0; k < layers_.size(); ++k) {
    int layer = k + 1;
    for (int l = 0; l < cassettes_; ++l) {
      int cassette = l + 1;
      auto cf1 = cassette_.getShift(layer, 1, cassette, false);
      auto cf2 = cassette_.getShift(layer, 1, cassette, true);
      auto cf3 = cassette_.getShift(layer, -1, cassette, false);
      auto cf4 = cassette_.getShift(layer, -1, cassette, true);
      edm::LogVerbatim("HGCalGeom") << "Layer " << layer << " Cassette " << cassette << " x for z+ " << cf1.first << ":"
                                    << cf2.first << " y for z+ " << cf1.second << ":" << cf2.second << " x for z- "
                                    << cf3.first << ":" << cf4.first << " y for z- " << cf3.second << ":" << cf4.second;
    }
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalMixRotatedFineCassette, "hgcal:DDHGCalMixRotatedFineCassette");
