///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalSiliconRotatedModule.cc
// Description: Geometry factory class for HGCal (EE and HESil) using
//              information from the file for dd4hep
///////////////////////////////////////////////////////////////////////////////
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "Geometry/HGCalCommonData/interface/HGCalCell.h"
#include "Geometry/HGCalCommonData/interface/HGCalCassette.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalProperty.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"
#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/angle_units.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DetectorDescription/DDCMS/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG
using namespace angle_units::operators;

struct HGCalSiliconRotatedModule {
  HGCalSiliconRotatedModule() {
    throw cms::Exception("HGCalGeom") << "Wrong initialization to HGCalSiliconRotatedModule";
  }
  HGCalSiliconRotatedModule(cms::DDParsingContext& ctxt, xml_h e) {
    cms::DDNamespace ns(ctxt, e, true);
    cms::DDAlgoArguments args(ctxt, e);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedModule: Creating an instance";
#endif
    static constexpr double tol1 = 0.01 * dd4hep::mm;
    static constexpr double tol2 = 0.00001 * dd4hep::mm;

    dd4hep::Volume mother = ns.volume(args.parentName());
    waferTypes_ = args.value<int>("WaferTypes");
    facingTypes_ = args.value<int>("FacingTypes");
    orientationTypes_ = args.value<int>("OrientationTypes");
    placeOffset_ = args.value<int>("PlaceOffset");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Number of types of wafers: " << waferTypes_ << " facings: " << facingTypes_
                                  << " Orientations: " << orientationTypes_ << " PlaceOffset: " << placeOffset_;
#endif
    firstLayer_ = args.value<int>("FirstLayer");
    absorbMode_ = args.value<int>("AbsorberMode");
    sensitiveMode_ = args.value<int>("SensitiveMode");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "First Layer " << firstLayer_ << " and "
                                  << "Absober:Sensitive mode " << absorbMode_ << ":" << sensitiveMode_;
#endif
    zMinBlock_ = args.value<double>("zMinBlock");
    waferSize_ = args.value<double>("waferSize");
    waferSepar_ = args.value<double>("SensorSeparation");
    sectors_ = args.value<int>("Sectors");
    cassettes_ = args.value<int>("Cassettes");
    alpha_ = (1._pi) / sectors_;
    cosAlpha_ = cos(alpha_);
    rotstr_ = args.value<std::string>("LayerRotation");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "zStart " << cms::convert2mm(zMinBlock_) << " wafer width "
                                  << cms::convert2mm(waferSize_) << " separations " << cms::convert2mm(waferSepar_)
                                  << " sectors " << sectors_ << ":" << convertRadToDeg(alpha_) << ":" << cosAlpha_
                                  << " rotation matrix " << rotstr_ << " with " << cassettes_ << " cassettes";
#endif
    waferFull_ = args.value<std::vector<std::string>>("WaferNamesFull");
    waferPart_ = args.value<std::vector<std::string>>("WaferNamesPartial");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedModule: " << waferFull_.size() << " full and "
                                  << waferPart_.size() << " partial modules\nDDHGCalSiliconRotatedModule:Full Modules:";
    unsigned int i1max = static_cast<unsigned int>(waferFull_.size());
    for (unsigned int i1 = 0; i1 < i1max; i1 += 2) {
      std::ostringstream st1;
      unsigned int i2 = std::min((i1 + 2), i1max);
      for (unsigned int i = i1; i < i2; ++i)
        st1 << " [" << i << "] " << waferFull_[i];
      edm::LogVerbatim("HGCalGeom") << st1.str();
    }
    edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedModule: Partial Modules:";
    i1max = static_cast<unsigned int>(waferPart_.size());
    for (unsigned int i1 = 0; i1 < i1max; i1 += 2) {
      std::ostringstream st1;
      unsigned int i2 = std::min((i1 + 2), i1max);
      for (unsigned int i = i1; i < i2; ++i)
        st1 << " [" << i << "] " << waferPart_[i];
      edm::LogVerbatim("HGCalGeom") << st1.str();
    }
#endif
    materials_ = args.value<std::vector<std::string>>("MaterialNames");
    names_ = args.value<std::vector<std::string>>("VolumeNames");
    thick_ = args.value<std::vector<double>>("Thickness");
    copyNumber_.resize(materials_.size(), 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedModule: " << materials_.size() << " types of volumes";
    for (unsigned int i = 0; i < names_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << names_[i] << " of thickness "
                                    << cms::convert2mm(thick_[i]) << " filled with " << materials_[i]
                                    << " first copy number " << copyNumber_[i];
#endif
    layers_ = args.value<std::vector<int>>("Layers");
    layerThick_ = args.value<std::vector<double>>("LayerThick");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "There are " << layers_.size() << " blocks";
    for (unsigned int i = 0; i < layers_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << cms::convert2mm(layerThick_[i])
                                    << " with " << layers_[i] << " layers";
#endif
    layerType_ = args.value<std::vector<int>>("LayerType");
    layerSense_ = args.value<std::vector<int>>("LayerSense");
    layerOrient_ = args.value<std::vector<int>>("LayerTypes");
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
    slopeB_ = args.value<std::vector<double>>("SlopeBottom");
    zFrontB_ = args.value<std::vector<double>>("ZFrontBottom");
    rMinFront_ = args.value<std::vector<double>>("RMinFront");
    slopeT_ = args.value<std::vector<double>>("SlopeTop");
    zFrontT_ = args.value<std::vector<double>>("ZFrontTop");
    rMaxFront_ = args.value<std::vector<double>>("RMaxFront");
#ifdef EDM_ML_DEBUG
    for (unsigned int i = 0; i < slopeB_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Bottom Block [" << i << "] Zmin " << cms::convert2mm(zFrontB_[i]) << " Rmin "
                                    << cms::convert2mm(rMinFront_[i]) << " Slope " << slopeB_[i];
    for (unsigned int i = 0; i < slopeT_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Top Block [" << i << "] Zmin " << cms::convert2mm(zFrontT_[i]) << " Rmax "
                                    << cms::convert2mm(rMaxFront_[i]) << " Slope " << slopeT_[i];
#endif
    waferIndex_ = args.value<std::vector<int>>("WaferIndex");
    waferProperty_ = args.value<std::vector<int>>("WaferProperties");
    waferLayerStart_ = args.value<std::vector<int>>("WaferLayerStart");
    cassetteShift_ = args.value<std::vector<double>>("CassetteShift");
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
    edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedModule: " << cassetteShift_.size()
                                  << " elements for cassette shifts";
    unsigned int j1max = cassetteShift_.size();
    for (unsigned int j1 = 0; j1 < j1max; j1 += 6) {
      std::ostringstream st1;
      unsigned int j2 = std::min((j1 + 6), j1max);
      for (unsigned int j = j1; j < j2; ++j)
        st1 << " [" << j << "] " << std::setw(9) << cassetteShift_[j];
      edm::LogVerbatim("HGCalGeom") << st1.str();
    }

    edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedModule: NameSpace " << ns.name();
#endif
    cassette_.setParameter(cassettes_, cassetteShift_);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalSiliconRotatedModule...";
    copies_.clear();
#endif

    double zi(zMinBlock_);
    int laymin(0);
    for (unsigned int i = 0; i < layers_.size(); ++i) {
      double zo = zi + layerThick_[i];
      double routF = HGCalGeomTools::radius(zi, zFrontT_, rMaxFront_, slopeT_);
      int laymax = laymin + layers_[i];
      double zz = zi;
      double thickTot(0);
      for (int ly = laymin; ly < laymax; ++ly) {
        int ii = layerType_[ly];
        int copy = copyNumber_[ii];
        double hthick = 0.5 * thick_[ii];
        double rinB = HGCalGeomTools::radius(zo - tol1, zFrontB_, rMinFront_, slopeB_);
        zz += hthick;
        thickTot += thick_[ii];

        std::string name = names_[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedModule: Layer " << ly << ":" << ii << " Front "
                                      << cms::convert2mm(zi) << ", " << cms::convert2mm(routF) << " Back "
                                      << cms::convert2mm(zo) << ", " << cms::convert2mm(rinB)
                                      << " superlayer thickness " << cms::convert2mm(layerThick_[i]);
#endif

        dd4hep::Material matter = ns.material(materials_[ii]);
        dd4hep::Volume glog;

        if (layerSense_[ly] < 1) {
          std::vector<double> pgonZ, pgonRin, pgonRout;
          double rmax = routF * cosAlpha_ - tol1;
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
              pgonRout[isec] = pgonRout[isec] * cosAlpha_ - tol1;
          }
          dd4hep::Solid solid = dd4hep::Polyhedra(sectors_, -alpha_, 2._pi, pgonZ, pgonRin, pgonRout);
          ns.addSolidNS(ns.prepend(name), solid);
          glog = dd4hep::Volume(solid.name(), solid, matter);
          ns.addVolumeNS(glog);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedModule: " << solid.name() << " polyhedra of "
                                        << sectors_ << " sectors covering " << convertRadToDeg(-alpha_) << ":"
                                        << convertRadToDeg(-alpha_ + 2._pi) << " with " << pgonZ.size()
                                        << " sections and filled with " << matter.name();
          for (unsigned int k = 0; k < pgonZ.size(); ++k)
            edm::LogVerbatim("HGCalGeom") << "[" << k << "] z " << cms::convert2mm(pgonZ[k]) << " R "
                                          << cms::convert2mm(pgonRin[k]) << ":" << cms::convert2mm(pgonRout[k]);
#endif
        } else {
          double rins =
              (sensitiveMode_ < 1) ? rinB : HGCalGeomTools::radius(zz + hthick - tol1, zFrontB_, rMinFront_, slopeB_);
          double routs =
              (sensitiveMode_ < 1) ? routF : HGCalGeomTools::radius(zz - hthick, zFrontT_, rMaxFront_, slopeT_);
          dd4hep::Solid solid = dd4hep::Tube(rins, routs, hthick, 0.0, 2._pi);
          ns.addSolidNS(ns.prepend(name), solid);
          glog = dd4hep::Volume(solid.name(), solid, matter);
          ns.addVolumeNS(glog);

#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedModule: " << solid.name() << " Tubs made of "
                                        << matter.name() << " of dimensions " << cms::convert2mm(rinB) << ":"
                                        << cms::convert2mm(rins) << ", " << cms::convert2mm(routF) << ":"
                                        << cms::convert2mm(routs) << ", " << cms::convert2mm(hthick)
                                        << ", 0.0, 360.0 and position " << glog.name() << " number " << copy << ":"
                                        << layerOrient_[copy - firstLayer_];
#endif
          positionSensitive(ctxt, e, glog, (copy - firstLayer_));
        }

        dd4hep::Position r1(0, 0, zz);
        dd4hep::Rotation3D rot;
#ifdef EDM_ML_DEBUG
        std::string rotName("Null");
#endif
        if ((layerSense_[ly] > 0) && (layerOrient_[copy - firstLayer_] == HGCalTypes::WaferCenterR)) {
          rot = ns.rotation(rotstr_);
#ifdef EDM_ML_DEBUG
          rotName = rotstr_;
#endif
        }
        mother.placeVolume(glog, copy, dd4hep::Transform3D(rot, r1));
        int inc = ((layerSense_[ly] > 0) && (facingTypes_ > 1)) ? 2 : 1;
        copyNumber_[ii] = copy + inc;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedModule: " << glog.name() << " number " << copy
                                      << " positioned in " << mother.name() << " at (0,0," << cms::convert2mm(zz)
                                      << ") with " << rotName << " rotation";
#endif
        zz += hthick;
      }  // End of loop over layers in a block
      zi = zo;
      laymin = laymax;
      // Make consistency check of all the partitions of the block
      if (std::abs(thickTot - layerThick_[i]) >= tol2) {
        if (thickTot > layerThick_[i]) {
          edm::LogError("HGCalGeom") << "Thickness of the partition " << cms::convert2mm(layerThick_[i])
                                     << " is smaller than " << cms::convert2mm(thickTot)
                                     << ": thickness of all its components **** ERROR ****";
        } else {
          edm::LogWarning("HGCalGeom") << "Thickness of the partition " << cms::convert2mm(layerThick_[i])
                                       << " does not match with " << cms::convert2mm(thickTot) << " of the components";
        }
      }
    }  // End of loop over blocks

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedModule: " << copies_.size()
                                  << " different wafer copy numbers";
    int k(0);
    for (std::unordered_set<int>::const_iterator itr = copies_.begin(); itr != copies_.end(); ++itr, ++k) {
      edm::LogVerbatim("HGCalGeom") << "Copy [" << k << "] : " << (*itr);
    }
    copies_.clear();
    edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalSiliconRotatedModule construction...";
#endif
  }

  void positionSensitive(cms::DDParsingContext& ctxt, xml_h e, const dd4hep::Volume& glog, int layer) {
    cms::DDNamespace ns(ctxt, e, true);
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
    edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedModule: " << glog.name() << " r " << cms::convert2mm(delx)
                                  << " R " << cms::convert2mm(dely) << " dy " << cms::convert2mm(dy) << " Shift "
                                  << cms::convert2mm(xyoff.first) << ":" << cms::convert2mm(xyoff.second)
                                  << " WaferSize " << cms::convert2mm((waferSize_ + waferSepar_)) << " index "
                                  << firstWafer << ":" << (lastWafer - 1);
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
      auto cshift = cassette_.getShift(layer + 1, 1, cassette);
      double xpos = xyoff.first + cshift.first + nc * delx;
      double ypos = xyoff.second + cshift.second + nr * dy;
      std::string wafer;
      int i(999);
      if (part == HGCalTypes::WaferFull) {
        i = type * facingTypes_ * orientationTypes_ + place - placeOffset_;
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
      edm::LogVerbatim("HGCalGeom") << " DDHGCalSiliconRotatedModule: Layer"
                                    << HGCalWaferIndex::waferLayer(waferIndex_[k]) << " Wafer " << wafer << " number "
                                    << copy << " type:part:orien:ind " << type << ":" << part << ":" << orien << ":"
                                    << i << " layer:u:v:indx " << (layer + firstLayer_) << ":" << u << ":" << v;
      if (iu > ium)
        ium = iu;
      if (iv > ivm)
        ivm = iv;
      kount++;
      if (copies_.count(copy) == 0)
        copies_.insert(copy);
#endif
      dd4hep::Position tran(xpos, ypos, 0.0);
      glog.placeVolume(ns.volume(wafer), copy, tran);
#ifdef EDM_ML_DEBUG
      ++ntype[type];
      edm::LogVerbatim("HGCalGeom") << " DDHGCalSiliconRotatedModule: " << wafer << " number " << copy << " type "
                                    << layertype << ":" << type << " positioned in " << glog.name() << " at ("
                                    << cms::convert2mm(xpos) << "," << cms::convert2mm(ypos) << ",0) with no rotation";
#endif
    }

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalSiliconRotatedModule: Maximum # of u " << ium << " # of v " << ivm
                                  << " and " << kount << " wafers (" << ntype[0] << ":" << ntype[1] << ":" << ntype[2]
                                  << ") for " << glog.name();
#endif
  }

  //Required data members to cache the values from XML file
  HGCalGeomTools geomTools_;
  HGCalCassette cassette_;

  int waferTypes_;                      // Number of wafer types
  int facingTypes_;                     // Types of facings of modules toward IP
  int orientationTypes_;                // Number of wafer orienations
  int placeOffset_;                     // Offset for placement
  int firstLayer_;                      // Copy # of the first sensitive layer
  int absorbMode_;                      // Absorber mode
  int sensitiveMode_;                   // Sensitive mode
  double zMinBlock_;                    // Starting z-value of the block
  double waferSize_;                    // Width of the wafer
  double waferSepar_;                   // Sensor separation
  int sectors_;                         // Sectors
  int cassettes_;                       // Cassettes
  std::string rotstr_;                  // Rotation matrix (if needed)
  std::vector<std::string> waferFull_;  // Names of full wafer modules
  std::vector<std::string> waferPart_;  // Names of partial wafer modules
  std::vector<std::string> materials_;  // names of materials
  std::vector<std::string> names_;      // Names of volumes
  std::vector<double> thick_;           // Thickness of the material
  std::vector<int> copyNumber_;         // Initial copy numbers
  std::vector<int> layers_;             // Number of layers in a section
  std::vector<double> layerThick_;      // Thickness of each section
  std::vector<int> layerType_;          // Type of the layer
  std::vector<int> layerSense_;         // Content of a layer (sensitive?)
  std::vector<double> slopeB_;          // Slope at the lower R
  std::vector<double> zFrontB_;         // Starting Z values for the slopes
  std::vector<double> rMinFront_;       // Corresponding rMin's
  std::vector<double> slopeT_;          // Slopes at the larger R
  std::vector<double> zFrontT_;         // Starting Z values for the slopes
  std::vector<double> rMaxFront_;       // Corresponding rMax's
  std::vector<int> layerOrient_;        // Layer orientation (Centering, rotations..)
  std::vector<int> waferIndex_;         // Wafer index for the types
  std::vector<int> waferProperty_;      // Wafer property
  std::vector<int> waferLayerStart_;    // Index of wafers in each layer
  std::vector<double> cassetteShift_;   // Shifts of the cassetes
  std::unordered_set<int> copies_;      // List of copy #'s
  double alpha_, cosAlpha_;
};

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  HGCalSiliconRotatedModule eealgo(ctxt, e);
  return cms::s_executed;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalSiliconRotatedModule, algorithm)
