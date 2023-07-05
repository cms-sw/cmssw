///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalSiliconRotatedCassette.cc
// Description: Geometry factory class for HGCal (EE and HESil) using
//              information from the file
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
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"

#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#define EDM_ML_DEBUG
using namespace angle_units::operators;

class DDHGCalSiliconRotatedCassette : public DDAlgorithm {
public:
  DDHGCalSiliconRotatedCassette();

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

protected:
  void constructLayers(const DDLogicalPart&, DDCompactView& cpv);
  void positionSensitive(const DDLogicalPart& glog, int layer, DDCompactView& cpv);
  void positionPassive(const DDLogicalPart& glog, int layer, int passiveType, DDCompactView& cpv);

private:
  HGCalGeomTools geomTools_;
  HGCalCassette cassette_;

  static constexpr double tol1_ = 0.01;
  static constexpr double tol2_ = 0.00001;

  int waferTypes_;                        // Number of wafer types
  int passiveTypes_;                      // Number of passive types
  int facingTypes_;                       // Types of facings of modules toward IP
  int orientationTypes_;                  // Number of wafer orienations
  int partialTypes_;                      // Number of partial types
  int placeOffset_;                       // Offset for placement
  int firstLayer_;                        // Copy # of the first sensitive layer
  int absorbMode_;                        // Absorber mode
  int sensitiveMode_;                     // Sensitive mode
  double zMinBlock_;                      // Starting z-value of the block
  double waferSize_;                      // Width of the wafer
  double waferSepar_;                     // Sensor separation
  int sectors_;                           // Sectors
  int cassettes_;                         // Cassettes
  std::string rotstr_;                    // Rotation matrix (if needed)
  std::vector<std::string> waferFull_;    // Names of full wafer modules
  std::vector<std::string> waferPart_;    // Names of partial wafer modules
  std::vector<std::string> passiveFull_;  // Names of full passive modules
  std::vector<std::string> passivePart_;  // Names of partial passive modules
  std::vector<std::string> materials_;    // names of materials
  std::vector<std::string> names_;        // Names of volumes
  std::vector<double> thick_;             // Thickness of the material
  std::vector<int> copyNumber_;           // Initial copy numbers
  std::vector<int> layers_;               // Number of layers in a section
  std::vector<double> layerThick_;        // Thickness of each section
  std::vector<int> layerType_;            // Type of the layer
  std::vector<int> layerSense_;           // Content of a layer (sensitive?)
  std::vector<double> slopeB_;            // Slope at the lower R
  std::vector<double> zFrontB_;           // Starting Z values for the slopes
  std::vector<double> rMinFront_;         // Corresponding rMin's
  std::vector<double> slopeT_;            // Slopes at the larger R
  std::vector<double> zFrontT_;           // Starting Z values for the slopes
  std::vector<double> rMaxFront_;         // Corresponding rMax's
  std::vector<int> layerOrient_;          // Layer orientation (Centering, rotations..)
  std::vector<int> waferIndex_;           // Wafer index for the types
  std::vector<int> waferProperty_;        // Wafer property
  std::vector<int> waferLayerStart_;      // Index of wafers in each layer
  std::vector<double> cassetteShift_;     // Shifts of the cassetes
  std::string nameSpace_;                 // Namespace of this and ALL sub-parts
  std::unordered_set<int> copies_;        // List of copy #'s
  double alpha_, cosAlpha_;
};

DDHGCalSiliconRotatedCassette::DDHGCalSiliconRotatedCassette() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: Creating an instance";
#endif
}

void DDHGCalSiliconRotatedCassette::initialize(const DDNumericArguments& nArgs,
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
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Number of types of wafers: " << waferTypes_ << " passives: " << passiveTypes_
                                << " facings: " << facingTypes_ << " Orientations: " << orientationTypes_
                                << " PartialTypes: " << partialTypes_ << " PlaceOffset: " << placeOffset_;
#endif
  firstLayer_ = static_cast<int>(nArgs["FirstLayer"]);
  absorbMode_ = static_cast<int>(nArgs["AbsorberMode"]);
  sensitiveMode_ = static_cast<int>(nArgs["SensitiveMode"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "First Layer " << firstLayer_ << " and "
                                << "Absober:Sensitive mode " << absorbMode_ << ":" << sensitiveMode_;
#endif
  zMinBlock_ = nArgs["zMinBlock"];
  waferSize_ = nArgs["waferSize"];
  waferSepar_ = nArgs["SensorSeparation"];
  sectors_ = static_cast<int>(nArgs["Sectors"]);
  cassettes_ = static_cast<int>(nArgs["Cassettes"]);
  alpha_ = (1._pi) / sectors_;
  cosAlpha_ = cos(alpha_);
  rotstr_ = sArgs["LayerRotation"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "zStart " << zMinBlock_ << " wafer width " << waferSize_ << " separations "
                                << waferSepar_ << " sectors " << sectors_ << ":" << convertRadToDeg(alpha_) << ":"
                                << cosAlpha_ << " rotation matrix " << rotstr_ << " with " << cassettes_
                                << " cassettes";
#endif
  waferFull_ = vsArgs["WaferNamesFull"];
  waferPart_ = vsArgs["WaferNamesPartial"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: " << waferFull_.size() << " full and "
                                << waferPart_.size() << " partial modules";
  unsigned int i1max = static_cast<unsigned int>(waferFull_.size());
  for (unsigned int i1 = 0; i1 < i1max; i1 += 2) {
    std::ostringstream st1;
    unsigned int i2 = std::min((i1 + 2), i1max);
    for (unsigned int i = i1; i < i2; ++i)
      st1 << " [" << i << "] " << waferFull_[i];
    edm::LogVerbatim("HGCalGeom") << st1.str();
  }
  edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: Partial Modules:";
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
  edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: " << materials_.size() << " types of volumes";
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
    edm::LogVerbatim("HGCalGeom") << "LayerTypes [" << i << "] " << layerOrient_[i];
#endif
  if (firstLayer_ > 0) {
    for (unsigned int i = 0; i < layerType_.size(); ++i) {
      if (layerSense_[i] > 0) {
        int ii = layerType_[i];
        copyNumber_[ii] = (layerSense_[i] == 1) ? firstLayer_ : (firstLayer_ + 1);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "First copy number for layer type " << i << ":" << ii << " with "
                                      << materials_[ii] << " changed to " << copyNumber_[ii];
#endif
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
    edm::LogVerbatim("HGCalGeom") << "Wafer[" << k << "] " << waferIndex_[k] << " ("
                                  << HGCalWaferIndex::waferLayer(waferIndex_[k]) << ", "
                                  << HGCalWaferIndex::waferU(waferIndex_[k]) << ", "
                                  << HGCalWaferIndex::waferV(waferIndex_[k]) << ") : ("
                                  << HGCalProperty::waferThick(waferProperty_[k]) << ":"
                                  << HGCalProperty::waferPartial(waferProperty_[k]) << ":"
                                  << HGCalProperty::waferOrient(waferProperty_[k]) << ")";
  edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: " << cassetteShift_.size()
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
  nameSpace_ = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: NameSpace " << nameSpace_ << ":";
#endif
  cassette_.setParameter(cassettes_, cassetteShift_);
}

////////////////////////////////////////////////////////////////////
// DDHGCalSiliconRotatedCassette methods...
////////////////////////////////////////////////////////////////////

void DDHGCalSiliconRotatedCassette::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalSiliconRotatedCassette...";
  copies_.clear();
#endif
  constructLayers(parent(), cpv);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: " << copies_.size()
                                << " different wafer copy numbers";
  int k(0);
  for (std::unordered_set<int>::const_iterator itr = copies_.begin(); itr != copies_.end(); ++itr, ++k) {
    edm::LogVerbatim("HGCalGeom") << "Copy [" << k << "] : " << (*itr);
  }
  copies_.clear();
  edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalSiliconRotatedCassette construction...";
#endif
}

void DDHGCalSiliconRotatedCassette::constructLayers(const DDLogicalPart& module, DDCompactView& cpv) {
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
      double rinB = HGCalGeomTools::radius(zo - tol1_, zFrontB_, rMinFront_, slopeB_);
      zz += hthick;
      thickTot += thick_[ii];

      std::string name = names_[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: Layer " << ly << ":" << ii << " Front " << zi
                                    << ", " << routF << " Back " << zo << ", " << rinB << " superlayer thickness "
                                    << layerThick_[i];
#endif
      DDName matName(DDSplit(materials_[ii]).first, DDSplit(materials_[ii]).second);
      DDMaterial matter(matName);
      DDLogicalPart glog;
      if (layerSense_[ly] == 0) {
        std::vector<double> pgonZ, pgonRin, pgonRout;
        double rmax = routF * cosAlpha_ - tol1_;
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
        edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: " << solid.name() << " polyhedra of "
                                      << sectors_ << " sectors covering " << convertRadToDeg(-alpha_) << ":"
                                      << convertRadToDeg(-alpha_ + 2._pi) << " with " << pgonZ.size()
                                      << " sections and filled with " << matName;
        for (unsigned int k = 0; k < pgonZ.size(); ++k)
          edm::LogVerbatim("HGCalGeom") << "[" << k << "] z " << pgonZ[k] << " R " << pgonRin[k] << ":" << pgonRout[k];
#endif
      } else {
        int mode = (layerSense_[ly] > 0) ? sensitiveMode_ : absorbMode_;
        double rins = (mode < 1) ? rinB : HGCalGeomTools::radius(zz + hthick - tol1_, zFrontB_, rMinFront_, slopeB_);
        double routs = (mode < 1) ? routF : HGCalGeomTools::radius(zz - hthick, zFrontT_, rMaxFront_, slopeT_);
        DDSolid solid = DDSolidFactory::tubs(DDName(name, nameSpace_), hthick, rins, routs, 0.0, 2._pi);
        glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: " << solid.name() << " Tubs made of "
                                      << matName << " of dimensions " << rinB << ":" << rins << ", " << routF << ":"
                                      << routs << ", " << hthick << ", 0.0, 360.0 and position " << glog.name()
                                      << " number " << copy << ":" << layerOrient_[copy - firstLayer_] << " Z " << zz;
#endif
        if (layerSense_[ly] > 0)
          positionSensitive(glog, (copy - firstLayer_), cpv);
        else
          positionPassive(glog, (copy - firstLayer_), -layerSense_[ly], cpv);
      }
      DDTranslation r1(0, 0, zz);
      DDRotation rot;
#ifdef EDM_ML_DEBUG
      std::string rotName("Null");
#endif
      if ((layerSense_[ly] != 0) && (layerOrient_[copy - firstLayer_] == HGCalTypes::WaferCenterR)) {
        rot = DDRotation(DDName(DDSplit(rotstr_).first, DDSplit(rotstr_).second));
#ifdef EDM_ML_DEBUG
        rotName = rotstr_;
#endif
      }
      cpv.position(glog, module, copy, r1, rot);
      int inc = ((layerSense_[ly] > 0) && (facingTypes_ > 1)) ? 2 : 1;
      copyNumber_[ii] = copy + inc;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: " << glog.name() << " number " << copy
                                    << " positioned in " << module.name() << " at " << r1 << " with " << rotName
                                    << " rotation";
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

// Position the silicon modules
void DDHGCalSiliconRotatedCassette::positionSensitive(const DDLogicalPart& glog, int layer, DDCompactView& cpv) {
  static const double sqrt3 = std::sqrt(3.0);
  int layercenter = layerOrient_[layer];
  int layertype = (layerOrient_[layer] == HGCalTypes::WaferCenterB) ? 1 : 0;
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
  edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: " << glog.ddname() << "  r " << delx << " R " << dely
                                << " dy " << dy << " Shift " << xyoff.first << ":" << xyoff.second << " WaferSize "
                                << (waferSize_ + waferSepar_) << " index " << firstWafer << ":" << (lastWafer - 1)
                                << " Layer Center " << layercenter << ":" << layertype;
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
    auto cshift = cassette_.getShift(layer + 1, -1, cassette);
    double xpos = xyoff.first - cshift.first + nc * delx;
    double ypos = xyoff.second + cshift.second + nr * dy;
#ifdef EDM_ML_DEBUG
    double xorig = xyoff.first + nc * delx;
    double yorig = xyoff.second + nr * dy;
    double angle = std::atan2(yorig, xorig);
    edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette::Wafer: layer " << layer + 1 << " cassette "
                                  << cassette << " Shift " << cshift.first << ":" << cshift.second << " Original "
                                  << xorig << ":" << yorig << ":" << convertRadToDeg(angle) << " Final " << xpos << ":"
                                  << ypos << " u|v " << u << ":" << v << " type|part|orient|place " << type << ":"
                                  << part << ":" << orien << ":" << place;
#endif
    std::string wafer;
    int i(999);
    if (part == HGCalTypes::WaferFull) {
      i = type * facingTypes_ * orientationTypes_ + place - placeOffset_;
      wafer = waferFull_[i];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << " layertype:type:part:orien:cassette:place:offsets:ind " << layertype << ":"
                                    << type << ":" << part << ":" << orien << ":" << cassette << ":" << place << ":"
                                    << placeOffset_ << ":" << facingTypes_ << ":" << orientationTypes_ << " wafer " << i
                                    << ":" << wafer;
#endif
    } else {
      int partoffset = (part >= HGCalTypes::WaferHDTop) ? HGCalTypes::WaferPartHDOffset : HGCalTypes::WaferPartLDOffset;
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
    edm::LogVerbatim("HGCalGeom") << " DDHGCalSiliconRotatedCassette: Layer "
                                  << HGCalWaferIndex::waferLayer(waferIndex_[k]) << " Wafer " << wafer << " number "
                                  << copy << " type:part:orien:place:ind " << type << ":" << part << ":" << orien << ":" << place << ":" << i
                                  << " layer:u:v:indx " << (layer + firstLayer_) << ":" << u << ":" << v << " pos " << xpos << ":" << ypos;
    if (iu > ium)
      ium = iu;
    if (iv > ivm)
      ivm = iv;
    kount++;
    if (copies_.count(copy) == 0)
      copies_.insert(copy);
#endif
    DDTranslation tran(xpos, ypos, 0.0);
    DDRotation rotation;
    DDName name = DDName(DDSplit(wafer).first, DDSplit(wafer).second);
    cpv.position(name, glog.ddname(), copy, tran, rotation);
#ifdef EDM_ML_DEBUG
    ++ntype[type];
    edm::LogVerbatim("HGCalGeom") << " DDHGCalSiliconRotatedCassette: " << name << " number " << copy << " type "
                                  << layertype << ":" << type << " positioned in " << glog.ddname() << " at " << tran
                                  << " with no rotation";
#endif
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: Maximum # of u " << ium << " # of v " << ivm
                                << " and " << kount << " passives (" << ntype[0] << ":" << ntype[1] << ":" << ntype[2]
                                << ") for " << glog.ddname();
#endif
}

// Position the passive modules
void DDHGCalSiliconRotatedCassette::positionPassive(const DDLogicalPart& glog,
                                                    int layer,
                                                    int absType,
                                                    DDCompactView& cpv) {
  static const double sqrt3 = std::sqrt(3.0);
  int layercenter = layerOrient_[layer];
  int layertype = (layerOrient_[layer] == HGCalTypes::WaferCenterB) ? 1 : 0;
  int firstWafer = waferLayerStart_[layer];
  int lastWafer = ((layer + 1 < static_cast<int>(waferLayerStart_.size())) ? waferLayerStart_[layer + 1]
                                                                           : static_cast<int>(waferIndex_.size()));
  double delx = 0.5 * (waferSize_ + waferSepar_);
  double dely = 2.0 * delx / sqrt3;
  double dy = 0.75 * dely;
  const auto& xyoff = geomTools_.shiftXY(layercenter, (waferSize_ + waferSepar_));
#ifdef EDM_ML_DEBUG
  int ium(0), ivm(0), kount(0);
  edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: " << glog.ddname() << "  r " << delx << " R " << dely
                                << " dy " << dy << " Shift " << xyoff.first << ":" << xyoff.second << " WaferSize "
                                << (waferSize_ + waferSepar_) << " index " << firstWafer << ":" << (lastWafer - 1)
                                << " Layer Center " << layercenter << ":" << layertype;
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
    auto cshift = cassette_.getShift(layer + 1, -1, cassette);
    double xpos = xyoff.first - cshift.first + nc * delx;
    double ypos = xyoff.second + cshift.second + nr * dy;
#ifdef EDM_ML_DEBUG
    double xorig = xyoff.first + nc * delx;
    double yorig = xyoff.second + nr * dy;
    double angle = std::atan2(yorig, xorig);
    edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette::Passive: layer " << layer + 1 << " cassette "
                                  << cassette << " Shift " << cshift.first << ":" << cshift.second << " Original "
                                  << xorig << ":" << yorig << ":" << convertRadToDeg(angle) << " Final " << xpos << ":"
                                  << ypos << " u|v " << u << ":" << v << " type|part|orient" << type << ":" << part
                                  << ":" << orien;
#endif
    std::string passive;
    int i(999);
    if (part == HGCalTypes::WaferFull) {
      i = absType - 1;
      passive = passiveFull_[i];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << " layertype:abstype:part:orien:cassette:offsets:ind " << layertype << ":"
                                    << absType << ":" << part << ":" << orien << ":" << cassette << ":"
                                    << ":" << partialTypes_ << ":" << orientationTypes_ << " passive " << i << ":"
                                    << passive;
#endif
    } else {
      int partoffset = (part >= HGCalTypes::WaferHDTop) ? HGCalTypes::WaferPartHDOffset : (HGCalTypes::WaferPartLDOffset - HGCalTypes::WaferTypeOffset[1]);
      i = (part - partoffset) * facingTypes_ * orientationTypes_ +
          (absType - 1) * facingTypes_ * orientationTypes_ * partialTypes_ + place - placeOffset_;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << " layertype:abstype:part:orien:cassette:3Types:offset:ind " << layertype << ":"
                                    << absType << ":" << part << ":" << orien << ":" << cassette << ":" << partialTypes_
                                    << ":" << facingTypes_ << ":" << orientationTypes_ << ":" << partoffset << ":" << i << ":"
                                    << passivePart_.size();
#endif
      passive = passivePart_[i];
    }
    int copy = HGCalTypes::packTypeUV(absType, u, v);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << " DDHGCalSiliconRotatedCassette: Layer "
                                  << HGCalWaferIndex::waferLayer(waferIndex_[k]) << " Passive " << passive << " number "
                                  << copy << " type:part:orien:place:ind " << type << ":" << part << ":" << orien << ":" << place << ":" << i
                                  << " layer:u:v:indx " << (layer + firstLayer_) << ":" << u << ":" << v << " pos " << xpos << ":" << ypos;
    if (iu > ium)
      ium = iu;
    if (iv > ivm)
      ivm = iv;
    kount++;
#endif
    DDTranslation tran(xpos, ypos, 0.0);
    DDRotation rotation;
    DDName name = DDName(DDSplit(passive).first, DDSplit(passive).second);
    cpv.position(name, glog.ddname(), copy, tran, rotation);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << " DDHGCalSiliconRotatedCassette: " << name << " number " << copy << " type "
                                  << layertype << ":" << type << " positioned in " << glog.ddname() << " at " << tran
                                  << " with no rotation";
#endif
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedCassette: Maximum # of u " << ium << " # of v " << ivm
                                << " and " << kount << " passives for " << glog.ddname();
#endif
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalSiliconRotatedCassette, "hgcal:DDHGCalSiliconRotatedCassette");
