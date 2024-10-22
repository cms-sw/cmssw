///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalEEFileAlgo.cc
// Description: Geometry factory class for HGCal (EE and HESil) using
//              information from the file for dd4hep
///////////////////////////////////////////////////////////////////////////////
#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

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

struct HGCalEEFileAlgo {
  HGCalEEFileAlgo() { throw cms::Exception("HGCalGeom") << "Wrong initialization to HGCalEEFileAlgo"; }
  HGCalEEFileAlgo(cms::DDParsingContext& ctxt, xml_h e) {
    cms::DDNamespace ns(ctxt, e, true);
    cms::DDAlgoArguments args(ctxt, e);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: Creating an instance";
#endif
    static constexpr double tol1 = 0.01 * dd4hep::mm;
    static constexpr double tol2 = 0.00001 * dd4hep::mm;

    dd4hep::Volume mother = ns.volume(args.parentName());
    wafers_ = args.value<std::vector<std::string>>("WaferNames");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << wafers_.size() << " wafers";
    for (unsigned int i = 0; i < wafers_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << "] " << wafers_[i];
#endif
    materials_ = args.value<std::vector<std::string>>("MaterialNames");
    names_ = args.value<std::vector<std::string>>("VolumeNames");
    thick_ = args.value<std::vector<double>>("Thickness");
    copyNumber_.resize(materials_.size(), 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << materials_.size() << " types of volumes";
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
    firstLayer_ = args.value<int>("FirstLayer");
    absorbMode_ = args.value<int>("AbsorberMode");
    sensitiveMode_ = args.value<int>("SensitiveMode");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "First Layer " << firstLayer_ << " and "
                                  << "Absober:Sensitive mode " << absorbMode_ << ":" << sensitiveMode_;
#endif
    layerCenter_ = args.value<std::vector<int>>("LayerCenter");
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
    zMinBlock_ = args.value<double>("zMinBlock");
    waferSize_ = args.value<double>("waferSize");
    waferSepar_ = args.value<double>("SensorSeparation");
    sectors_ = args.value<int>("Sectors");
    alpha_ = (1._pi) / sectors_;
    cosAlpha_ = cos(alpha_);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "zStart " << cms::convert2mm(zMinBlock_) << " wafer width "
                                  << cms::convert2mm(waferSize_) << " separations " << cms::convert2mm(waferSepar_)
                                  << " sectors " << sectors_ << ":" << convertRadToDeg(alpha_) << ":" << cosAlpha_;
#endif
    waferIndex_ = args.value<std::vector<int>>("WaferIndex");
    waferProperty_ = args.value<std::vector<int>>("WaferProperties");
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
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: NameSpace " << ns.name();
#endif

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalEEFileAlgo...";
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
        edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: Layer " << ly << ":" << ii << " Front "
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
          edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << solid.name() << " polyhedra of " << sectors_
                                        << " sectors covering " << convertRadToDeg(-alpha_) << ":"
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
          edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << solid.name() << " Tubs made of " << matter.name()
                                        << " of dimensions " << cms::convert2mm(rinB) << ":" << cms::convert2mm(rins)
                                        << ", " << cms::convert2mm(routF) << ":" << cms::convert2mm(routs) << ", "
                                        << cms::convert2mm(hthick) << ", 0.0, 360.0 and position " << glog.name()
                                        << " number " << copy << ":" << layerCenter_[copy - firstLayer_];
#endif
          positionSensitive(ctxt, e, glog, rins, routs, zz, layerSense_[ly], (copy - firstLayer_));
        }

        dd4hep::Position r1(0, 0, zz);
        mother.placeVolume(glog, copy, r1);
        ++copyNumber_[ii];
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << glog.name() << " number " << copy << " positioned in "
                                      << mother.name() << " at (0,0," << cms::convert2mm(zz) << ") with no rotation";
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
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << copies_.size() << " different wafer copy numbers";
    int k(0);
    for (std::unordered_set<int>::const_iterator itr = copies_.begin(); itr != copies_.end(); ++itr, ++k) {
      edm::LogVerbatim("HGCalGeom") << "Copy [" << k << "] : " << (*itr);
    }
    copies_.clear();
    edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalEEFileAlgo construction...";
#endif
  }

  void positionSensitive(cms::DDParsingContext& ctxt,
                         xml_h e,
                         const dd4hep::Volume& glog,
                         double rin,
                         double rout,
                         double zpos,
                         int layertype,
                         int layer) {
    cms::DDNamespace ns(ctxt, e, true);
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
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << glog.name() << " rin:rout " << cms::convert2mm(rin) << ":"
                                  << cms::convert2mm(rout) << " zpos " << cms::convert2mm(zpos) << " N " << N
                                  << " for maximum u, v;  r " << cms::convert2mm(r) << " R " << cms::convert2mm(R)
                                  << " dy " << cms::convert2mm(dy) << " Shift " << cms::convert2mm(xyoff.first) << ":"
                                  << cms::convert2mm(xyoff.second) << " WaferSize "
                                  << cms::convert2mm((waferSize_ + waferSepar_));
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
        if (((corner.first <= 0) && std::abs(u) < 5 && std::abs(v) < 5) || (std::abs(u) < 2 && std::abs(v) < 2)) {
          edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: " << glog.name() << " R " << cms::convert2mm(rin) << ":"
                                        << cms::convert2mm(rout) << "\n Z " << cms::convert2mm(zpos) << " LayerType "
                                        << layertype << " u " << u << " v " << v << " with " << corner.first
                                        << " corners";
        }
#endif
        int indx = HGCalWaferIndex::waferIndex((layer + firstLayer_), u, v, false);
        int type = HGCalWaferType::getType(indx, waferIndex_, waferProperty_);
        if (corner.first > 0 && type >= 0) {
          int copy = HGCalTypes::packTypeUV(type, u, v);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << " DDHGCalEEFileAlgo: " << wafers_[type] << " number " << copy << " type "
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
            dd4hep::Position tran(xpos, ypos, 0.0);
            if (layertype > 1)
              type += 3;
            glog.placeVolume(ns.volume(wafers_[type]), copy, tran);
#ifdef EDM_ML_DEBUG
            ++ntype[type];
            edm::LogVerbatim("HGCalGeom")
                << " DDHGCalEEFileAlgo: " << wafers_[type] << " number " << copy << " type " << layertype << ":" << type
                << " positioned in " << glog.name() << " at (" << cms::convert2mm(xpos) << ", " << cms::convert2mm(ypos)
                << ", 0) with no rotation";
#endif
          }
        }
      }
    }

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEFileAlgo: Maximum # of u " << ium << ":" << iumAll << " # of v " << ivm
                                  << ":" << ivmAll << " and " << nin << ":" << kount << ":" << ntot << " wafers ("
                                  << ntype[0] << ":" << ntype[1] << ":" << ntype[2] << ":" << ntype[3] << ":"
                                  << ntype[4] << ":" << ntype[5] << ") for " << glog.name() << " R "
                                  << cms::convert2mm(rin) << ":" << cms::convert2mm(rout);
#endif
  }

  //Required data members to cache the values from XML file
  HGCalGeomTools geomTools_;

  std::vector<std::string> wafers_;     // Wafers
  std::vector<std::string> materials_;  // Materials
  std::vector<std::string> names_;      // Names
  std::vector<double> thick_;           // Thickness of the material
  std::vector<int> copyNumber_;         // Initial copy numbers
  std::vector<int> layers_;             // Number of layers in a section
  std::vector<double> layerThick_;      // Thickness of each section
  std::vector<int> layerType_;          // Type of the layer
  std::vector<int> layerSense_;         // Content of a layer (sensitive?)
  std::vector<int> layerCenter_;        // Centering of the wafers
  int firstLayer_;                      // Copy # of the first sensitive layer
  int absorbMode_;                      // Absorber mode
  int sensitiveMode_;                   // Sensitive mode
  double zMinBlock_;                    // Starting z-value of the block
  std::vector<int> waferIndex_;         // Wafer index for the types
  std::vector<int> waferProperty_;      // Wafer property
  double waferSize_;                    // Width of the wafer
  double waferSepar_;                   // Sensor separation
  int sectors_;                         // Sectors
  std::vector<double> slopeB_;          // Slope at the lower R
  std::vector<double> zFrontB_;         // Starting Z values for the slopes
  std::vector<double> rMinFront_;       // Corresponding rMin's
  std::vector<double> slopeT_;          // Slopes at the larger R
  std::vector<double> zFrontT_;         // Starting Z values for the slopes
  std::vector<double> rMaxFront_;       // Corresponding rMax's
  std::unordered_set<int> copies_;      // List of copy #'s
  double alpha_, cosAlpha_;
};

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  HGCalEEFileAlgo eeFileAlgo(ctxt, e);
  return cms::s_executed;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalEEFileAlgo, algorithm)
