/*
 * DD4hep_HGCalEEAlgo.cc
 *
 *  Created on: 27-Aug-2019
 *      Author: rsehgal
 *
 *  DD4hep code for, HGCalEEAlgo developed by Sunanda Banerjee
 */

#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"

//#define EDM_ML_DEBUG
using namespace cms_units::operators;

struct HGCalEEAlgo {
  HGCalGeomTools geomTools_;
  std::unique_ptr<HGCalWaferType> waferType_;
  dd4hep::Volume mother_;

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
  double zMinBlock_;                    // Starting z-value of the block
  std::vector<double> rad100to200_;     // Parameters for 120-200mum trans.
  std::vector<double> rad200to300_;     // Parameters for 200-300mum trans.
  double zMinRadPar_;                   // Minimum z for radius parametriz.
  int choiceType_;                      // Type of parametrization to be used
  int nCutRadPar_;                      // Cut off threshold for corners
  double fracAreaMin_;                  // Minimum fractional conatined area
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

  HGCalEEAlgo() = delete;

  HGCalEEAlgo(cms::DDParsingContext& ctxt, xml_h e) {
    cms::DDNamespace ns(ctxt, e, true);
    cms::DDAlgoArguments args(ctxt, e);

    mother_ = ns.volume(args.parentName());
    wafers_ = args.value<std::vector<std::string>>("WaferNames");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEAlgo: " << wafers_.size() << " wafers";
    for (unsigned int i = 0; i < wafers_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << "] " << wafers_[i];
#endif

    materials_ = args.value<std::vector<std::string>>("MaterialNames");
    names_ = args.value<std::vector<std::string>>("VolumeNames");
    thick_ = args.value<std::vector<double>>("Thickness");
    copyNumber_.resize(materials_.size(), 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEAlgo: " << materials_.size() << " types of volumes";
    for (unsigned int i = 0; i < names_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << names_[i] << " of thickness " << thick_[i]
                                    << " filled with " << materials_[i] << " first copy number " << copyNumber_[i];
#endif

    layers_ = args.value<std::vector<int>>("Layers");
    layerThick_ = args.value<std::vector<double>>("LayerThick");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "There are " << layers_.size() << " blocks";
    for (unsigned int i = 0; i < layers_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << layerThick_[i] << " with " << layers_[i]
                                    << " layers";
#endif

    layerType_ = args.value<std::vector<int>>("LayerType");
    layerSense_ = args.value<std::vector<int>>("LayerSense");
    firstLayer_ = args.value<int>("FirstLayer");
    absorbMode_ = args.value<int>("AbsorberMode");

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "First Layere " << firstLayer_ << " and "
                                  << "Absober mode " << absorbMode_;
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
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "There are " << layerType_.size() << " layers";
    for (unsigned int i = 0; i < layerType_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerType_[i] << " sensitive class "
                                    << layerSense_[i];
#endif
    zMinBlock_ = args.value<double>("zMinBlock");

    rad100to200_ = args.value<std::vector<double>>("rad100to200");
    rad200to300_ = args.value<std::vector<double>>("rad200to300");
    zMinRadPar_ = args.value<double>("zMinForRadPar");
    choiceType_ = args.value<int>("choiceType");
    nCutRadPar_ = args.value<int>("nCornerCut");
    fracAreaMin_ = args.value<double>("fracAreaMin");
    waferSize_ = args.value<double>("waferSize");
    waferSepar_ = args.value<double>("SensorSeparation");
    sectors_ = args.value<int>("Sectors");
    alpha_ = (1._pi) / sectors_;
    cosAlpha_ = cos(alpha_);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "zStart " << zMinBlock_ << " radius for wafer type separation uses "
                                  << rad100to200_.size() << " parameters; zmin " << zMinRadPar_ << " cutoff "
                                  << choiceType_ << ":" << nCutRadPar_ << ":" << fracAreaMin_ << " wafer width "
                                  << waferSize_ << " separations " << waferSepar_ << " sectors " << sectors_ << ":"
                                  << convertRadToDeg(alpha_) << ":" << cosAlpha_;
    for (unsigned int k = 0; k < rad100to200_.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] 100-200 " << rad100to200_[k] << " 200-300 " << rad200to300_[k];
#endif

    slopeB_ = args.value<std::vector<double>>("SlopeBottom");
    zFrontB_ = args.value<std::vector<double>>("ZFrontBottom");
    rMinFront_ = args.value<std::vector<double>>("RMinFront");
    slopeT_ = args.value<std::vector<double>>("SlopeTop");
    zFrontT_ = args.value<std::vector<double>>("ZFrontTop");
    rMaxFront_ = args.value<std::vector<double>>("RMaxFront");
#ifdef EDM_ML_DEBUG
    for (unsigned int i = 0; i < slopeB_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] Zmin " << zFrontB_[i] << " Rmin " << rMinFront_[i]
                                    << " Slope " << slopeB_[i];
    for (unsigned int i = 0; i < slopeT_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] Zmin " << zFrontT_[i] << " Rmax " << rMaxFront_[i]
                                    << " Slope " << slopeT_[i];
#endif

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEAlgo: NameSpace " << ns.name();
#endif

    waferType_ = std::make_unique<HGCalWaferType>(
        rad100to200_, rad200to300_, (waferSize_ + waferSepar_), zMinRadPar_, choiceType_, nCutRadPar_, fracAreaMin_);

    ConstructAlgo(ctxt, e);
  }

  void ConstructAlgo(cms::DDParsingContext& ctxt, xml_h e) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalEEAlgo...";
    copies_.clear();
#endif
    dd4hep::Volume par;
    ConstructLayers(par, ctxt, e);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEAlgo: " << copies_.size() << " different wafer copy numbers";
    int k(0);
    for (std::unordered_set<int>::const_iterator itr = copies_.begin(); itr != copies_.end(); ++itr, ++k) {
      edm::LogVerbatim("HGCalGeom") << "Copy [" << k << "] : " << (*itr);
    }
    copies_.clear();
    edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalEEAlgo construction...";
#endif
  }

  void ConstructLayers(const dd4hep::Volume module, cms::DDParsingContext& ctxt, xml_h e) {
    static constexpr double tol1 = 0.01;
    static constexpr double tol2 = 0.00001;
    cms::DDNamespace ns(ctxt, e, true);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEAlgo: \t\tInside Layers";
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

        std::string name = ns.prepend(names_[ii]) + std::to_string(copy);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalEEAlgo: Layer " << ly << ":" << ii << " Front " << zi << ", " << routF
                                      << " Back " << zo << ", " << rinB << " superlayer thickness " << layerThick_[i];
#endif

        std::string matName = materials_[ii];
        dd4hep::Material matter = ns.material(matName);
        dd4hep::Volume glog;
        if (layerSense_[ly] < 1) {
          std::vector<double> pgonZ, pgonRin, pgonRout;
          if (layerSense_[ly] == 0 || absorbMode_ == 0) {
            double rmax = routF * cosAlpha_ - tol1;
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
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom") << "DDHGCalEEAlgo: z " << (zz - hthick) << ":" << (zz + hthick) << " with "
                                          << pgonZ.size() << " palnes";
            for (unsigned int isec = 0; isec < pgonZ.size(); ++isec)
              edm::LogVerbatim("HGCalGeom")
                  << "[" << isec << "] z " << pgonZ[isec] << " R " << pgonRin[isec] << ":" << pgonRout[isec];
#endif
            for (unsigned int isec = 0; isec < pgonZ.size(); ++isec) {
              pgonZ[isec] -= zz;
              pgonRout[isec] = pgonRout[isec] * cosAlpha_ - tol1;
            }
          }

          dd4hep::Solid solid =
              dd4hep::Polyhedra(sectors_, -alpha_, 2. * cms_units::piRadians, pgonZ, pgonRin, pgonRout);
          ns.addSolidNS(ns.prepend(name), solid);
          glog = dd4hep::Volume(solid.name(), solid, matter);
          ns.addVolumeNS(glog);

#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalEEAlgo: " << solid.name() << " polyhedra of " << sectors_
                                        << " sectors covering " << convertRadToDeg(-alpha_) << ":"
                                        << convertRadToDeg(-alpha_ + 2._pi) << " with " << pgonZ.size()
                                        << " sections and filled with " << matName;

          for (unsigned int k = 0; k < pgonZ.size(); ++k)
            edm::LogVerbatim("HGCalGeom")
                << "[" << k << "] z " << pgonZ[k] << " R " << pgonRin[k] << ":" << pgonRout[k];
#endif
        } else {
          dd4hep::Solid solid = dd4hep::Tube(rinB, routF, hthick, 0.0, 2. * cms_units::piRadians);
          ns.addSolidNS(ns.prepend(name), solid);
          glog = dd4hep::Volume(solid.name(), solid, matter);
          ns.addVolumeNS(glog);

#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalEEAlgo: " << solid.name() << " Tubs made of " << matName
                                        << " of dimensions " << rinB << ", " << routF << ", " << hthick
                                        << ", 0.0, 360.0 and position " << glog.name() << " number " << copy << ":"
                                        << layerCenter_[copy - 1];
#endif
          PositionSensitive(ctxt, e, glog, rinB, routF, zz, layerSense_[ly], layerCenter_[copy - 1]);  //, cpv);
        }

        dd4hep::Position r1(0, 0, zz);
        mother_.placeVolume(glog, copy, r1);
        ++copyNumber_[ii];

#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalEEAlgo: " << glog.name() << " number " << copy << " positioned in "
                                      << module.name() << " at " << r1 << " with no rotation";
#endif
        zz += hthick;
      }  // End of loop over layers in a block
      zi = zo;
      laymin = laymax;
      if (std::abs(thickTot - layerThick_[i]) >= tol2) {
        if (thickTot > layerThick_[i]) {
          edm::LogError("HGCalGeom") << "Thickness of the partition " << layerThick_[i] << " is smaller than "
                                     << thickTot << ": thickness of all its components **** ERROR ****";
        } else {
          edm::LogWarning("HGCalGeom") << "Thickness of the partition " << layerThick_[i] << " does not match with "
                                       << thickTot << " of the components";
        }
      }

    }  // End of loop over layers in a block
  }

  void PositionSensitive(cms::DDParsingContext& ctxt,
                         xml_h e,
                         const dd4hep::Volume& glog,
                         double rin,
                         double rout,
                         double zpos,
                         int layertype,
                         int layercenter) {
    cms::DDNamespace ns(ctxt, e, true);
    static const double sqrt3 = std::sqrt(3.0);
    double r = 0.5 * (waferSize_ + waferSepar_);
    double R = 2.0 * r / sqrt3;
    double dy = 0.75 * R;
    int N = (int)(0.5 * rout / r) + 2;
    const auto& xyoff = geomTools_.shiftXY(layercenter, (waferSize_ + waferSepar_));
#ifdef EDM_ML_DEBUG
    int ium(0), ivm(0), iumAll(0), ivmAll(0), kount(0), ntot(0), nin(0);
    std::vector<int> ntype(6, 0);
    edm::LogVerbatim("HGCalGeom") << "DDHGCalEEAlgo: " << glog.name() << " rout " << rout << " N " << N
                                  << " for maximum u, v;  r " << r << " R " << R << " dy " << dy << " Shift "
                                  << xyoff.first << ":" << xyoff.second << " WaferSize " << (waferSize_ + waferSepar_);
#endif

    for (int u = -N; u <= N; ++u) {
      for (int v = -N; v <= N; ++v) {
        int nr = 2 * v;
        int nc = -2 * u + v;
        double xpos = xyoff.first + nc * r;
        double ypos = xyoff.second + nr * dy;
        const auto& corner = HGCalGeomTools::waferCorner(xpos, ypos, r, R, rin, rout, false);
#ifdef EDM_ML_DEBUG
        int iu = std::abs(u);
        int iv = std::abs(v);
        ++ntot;
        if (((corner.first <= 0) && std::abs(u) < 5 && std::abs(v) < 5) || (std::abs(u) < 2 && std::abs(v) < 2)) {
          edm::LogVerbatim("HGCalGeom") << "DDHGCalEEAlgo: " << glog.name() << " R " << rin << ":" << rout << "\n Z "
                                        << zpos << " LayerType " << layertype << " u " << u << " v " << v << " with "
                                        << corner.first << " corners";
        }
#endif
        if (corner.first > 0) {
          int type = waferType_->getType(xpos, ypos, zpos);
          int copy = HGCalTypes::packTypeUV(type, u, v);
#ifdef EDM_ML_DEBUG
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
            edm::LogVerbatim("HGCalGeom") << " DDHGCalEEAlgo: " << wafers_[type] << " number " << copy
                                          << " positioned in " << glog.name() << " at " << tran << " with no rotation";
#endif
          }
        }
      }
    }

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << " DDHGCalEEAlgo: Maximum # of u " << ium << ":" << iumAll << " # of v " << ivm
                                  << ":" << ivmAll << " and " << nin << ":" << kount << ":" << ntot << " wafers ("
                                  << ntype[0] << ":" << ntype[1] << ":" << ntype[2] << ":" << ntype[3] << ":"
                                  << ntype[4] << ":" << ntype[5] << ") for " << glog.name() << " R " << rin << ":"
                                  << rout;
#endif
  }
};

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  HGCalEEAlgo eealgo(ctxt, e);
  return cms::s_executed;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalEEAlgo, algorithm)
