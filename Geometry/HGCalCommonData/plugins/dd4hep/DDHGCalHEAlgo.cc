///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalHEAlgo.cc
// Description: Geometry factory class for HGCal (Mix)
// Author : Raman Sehgal
// DD4hep code for, HGCalHEAlgo, developed by Sunanda Banerjee
///////////////////////////////////////////////////////////////////////////////

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

struct HGCalHEAlgo {
  HGCalHEAlgo() { throw cms::Exception("HGCalGeom") << "Wrong initialization to HGCalHEAlgo"; }
  HGCalHEAlgo(cms::DDParsingContext& ctxt, xml_h e) {
    cms::DDNamespace ns(ctxt, e, true);
    cms::DDAlgoArguments args(ctxt, e);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: Creating an instance";
#endif

    dd4hep::Volume mother = ns.volume(args.parentName());
    waferNames_ = args.value<std::vector<std::string>>("WaferNames");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << waferNames_.size() << " wafers";
    for (unsigned int i = 0; i < waferNames_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << "] " << waferNames_[i];
#endif
    materials_ = args.value<std::vector<std::string>>("MaterialNames");
    volumeNames_ = args.value<std::vector<std::string>>("VolumeNames");
    thickness_ = args.value<std::vector<double>>("Thickness");
    copyNumber_.resize(materials_.size(), 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << materials_.size() << " types of volumes";
    for (unsigned int i = 0; i < volumeNames_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << volumeNames_[i] << " of thickness " << thickness_[i]
                                    << " filled with " << materials_[i] << " first copy number " << copyNumber_[i];
#endif
    layerNumbers_ = args.value<std::vector<int>>("Layers");
    layerThick_ = args.value<std::vector<double>>("LayerThick");
    rMixLayer_ = args.value<std::vector<double>>("LayerRmix");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "There are " << layerNumbers_.size() << " blocks";
    for (unsigned int i = 0; i < layerNumbers_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Block [" << i << "] of thickness " << layerThick_[i] << " Rmid "
                                    << rMixLayer_[i] << " with " << layerNumbers_[i] << " layers";
#endif
    layerType_ = args.value<std::vector<int>>("LayerType");
    layerSense_ = args.value<std::vector<int>>("LayerSense");
    firstLayer_ = args.value<int>("FirstLayer");
    absorbMode_ = args.value<int>("AbsorberMode");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "First Layer " << firstLayer_ << " and "
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
    materialsTop_ = args.value<std::vector<std::string>>("TopMaterialNames");
    namesTop_ = args.value<std::vector<std::string>>("TopVolumeNames");
    layerThickTop_ = args.value<std::vector<double>>("TopLayerThickness");
    layerTypeTop_ = args.value<std::vector<int>>("TopLayerType");
    copyNumberTop_.resize(materialsTop_.size(), 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << materialsTop_.size() << " types of volumes in the top part";
    for (unsigned int i = 0; i < materialsTop_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << namesTop_[i] << " of thickness " << layerThickTop_[i]
                                    << " filled with " << materialsTop_[i] << " first copy number "
                                    << copyNumberTop_[i];
    edm::LogVerbatim("HGCalGeom") << "There are " << layerTypeTop_.size() << " layers in the top part";
    for (unsigned int i = 0; i < layerTypeTop_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerTypeTop_[i];
#endif
    materialsBot_ = args.value<std::vector<std::string>>("BottomMaterialNames");
    namesBot_ = args.value<std::vector<std::string>>("BottomVolumeNames");
    layerTypeBot_ = args.value<std::vector<int>>("BottomLayerType");
    layerSenseBot_ = args.value<std::vector<int>>("BottomLayerSense");
    layerThickBot_ = args.value<std::vector<double>>("BottomLayerThickness");
    copyNumberBot_.resize(materialsBot_.size(), 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << materialsBot_.size()
                                  << " types of volumes in the bottom part";
    for (unsigned int i = 0; i < materialsBot_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Volume [" << i << "] " << namesBot_[i] << " of thickness " << layerThickBot_[i]
                                    << " filled with " << materialsBot_[i] << " first copy number "
                                    << copyNumberBot_[i];
    edm::LogVerbatim("HGCalGeom") << "There are " << layerTypeBot_.size() << " layers in the bottom part";
    for (unsigned int i = 0; i < layerTypeBot_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Layer [" << i << "] with material type " << layerTypeBot_[i]
                                    << " sensitive class " << layerSenseBot_[i];
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
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: zStart " << zMinBlock_ << " radius for wafer type separation uses "
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
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: NameSpace " << ns.name();
#endif

    waferType_ = std::make_unique<HGCalWaferType>(
        rad100to200_, rad200to300_, (waferSize_ + waferSepar_), zMinRadPar_, choiceType_, nCutRadPar_, fracAreaMin_);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalHEAlgo...";
    copies_.clear();
#endif

    double zi(zMinBlock_);
    int laymin(0);
    for (unsigned int i = 0; i < layerNumbers_.size(); i++) {
      double zo = zi + layerThick_[i];
      double routF = HGCalGeomTools::radius(zi, zFrontT_, rMaxFront_, slopeT_);
      int laymax = laymin + layerNumbers_[i];
      double zz = zi;
      double thickTot(0);
      for (int ly = laymin; ly < laymax; ++ly) {
        int ii = layerType_[ly];
        int copy = copyNumber_[ii];
        double hthick = 0.5 * thickness_[ii];
        double rinB = HGCalGeomTools::radius(zo, zFrontB_, rMinFront_, slopeB_);
        zz += hthick;
        thickTot += thickness_[ii];

        std::string name = volumeNames_[ii] + std::to_string(copy);

#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: Layer " << ly << ":" << ii << " Front " << zi << ", " << routF
                                      << " Back " << zo << ", " << rinB << " superlayer thickness " << layerThick_[i];
#endif

        dd4hep::Material matter = ns.material(materials_[ii]);
        dd4hep::Volume glog;

        if (layerSense_[ly] < 1) {
          std::vector<double> pgonZ, pgonRin, pgonRout;
          if (layerSense_[ly] == 0 || absorbMode_ == 0) {
            double rmax =
                (std::min(routF, HGCalGeomTools::radius(zz + hthick, zFrontT_, rMaxFront_, slopeT_)) * cosAlpha_) -
                tol1_;
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

          dd4hep::Solid solid =
              dd4hep::Polyhedra(sectors_, -alpha_, 2. * cms_units::piRadians, pgonZ, pgonRin, pgonRout);
          ns.addSolidNS(ns.prepend(name), solid);
          glog = dd4hep::Volume(solid.name(), solid, matter);
          ns.addVolumeNS(glog);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << solid.name() << " polyhedra of " << sectors_
                                        << " sectors covering " << convertRadToDeg(-alpha_) << ":"
                                        << convertRadToDeg(-alpha_ + 2._pi) << " with " << pgonZ.size() << " sections";
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
          edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << solid.name() << " Tubs made of " << matter.name()
                                        << " of dimensions " << rinB << ", " << routF << ", " << hthick
                                        << ", 0.0, 360.0 and positioned in: " << glog.name() << " number " << copy;
#endif
          positionMix(ctxt, e, glog, name, copy, thickness_[ii], matter, rinB, rMixLayer_[i], routF, zz);
        }

        dd4hep::Position r1(0, 0, zz);
        mother.placeVolume(glog, copy, r1);
        ++copyNumber_[ii];
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << glog.name() << " number " << copy << " positioned in "
                                      << mother.name() << " at " << r1 << " with no rotation";
#endif
        zz += hthick;
      }  // End of loop over layers in a block
      zi = zo;
      laymin = laymax;
      if (std::abs(thickTot - layerThick_[i]) >= tol2_) {
        if (thickTot > layerThick_[i]) {
          edm::LogError("HGCalGeom") << "Thickness of the partition " << layerThick_[i] << " is smaller than "
                                     << thickTot << ": thickness of all its components **** ERROR ****";
        } else {
          edm::LogWarning("HGCalGeom") << "Thickness of the partition " << layerThick_[i] << " does not match with "
                                       << thickTot << " of the components";
        }
      }
    }  // End of loop over blocks

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << copies_.size() << " different wafer copy numbers";
    int k(0);
    for (std::unordered_set<int>::const_iterator itr = copies_.begin(); itr != copies_.end(); ++itr, ++k) {
      edm::LogVerbatim("HGCalGeom") << "Copy [" << k << "] : " << (*itr);
    }
    copies_.clear();
    edm::LogVerbatim("HGCalGeom") << "<<== End of DDHGCalHEAlgo construction...";
#endif
  }

  void positionMix(cms::DDParsingContext& ctxt,
                   xml_h e,
                   const dd4hep::Volume& glog,
                   const std::string& nameM,
                   int copyM,
                   double thick,
                   const dd4hep::Material& matter,
                   double rin,
                   double rmid,
                   double rout,
                   double zz) {
    cms::DDNamespace ns(ctxt, e, true);

    dd4hep::Volume glog1;
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

    dd4hep::Solid solid = dd4hep::Tube(rmid, rout, hthick, 0.0, 2. * cms_units::piRadians);
    ns.addSolidNS(ns.prepend(name), solid);
    glog1 = dd4hep::Volume(solid.name(), solid, matter);
    ns.addVolumeNS(glog1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << solid.name() << " Tubs made of " << matter.name()
                                  << " of dimensions " << rmid << ", " << rout << ", " << hthick << ", 0.0, 360.0";
#endif
    glog.placeVolume(glog1, 1);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << glog1.name() << " number 1 positioned in " << glog.name()
                                  << " at (0, 0, 0) with no rotation";
#endif
    double thickTot(0), zpos(-hthick);
    for (unsigned int ly = 0; ly < layerTypeTop_.size(); ++ly) {
      int ii = layerTypeTop_[ly];
      int copy = copyNumberTop_[ii];
      double hthickl = 0.5 * layerThickTop_[ii];
      thickTot += layerThickTop_[ii];
      name = namesTop_[ii] + std::to_string(copy);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: Layer " << ly << ":" << ii << " R " << rmid << ":" << rout
                                    << " Thick " << layerThickTop_[ii];
#endif

      dd4hep::Material matter1 = ns.material(materialsTop_[ii]);
      solid = dd4hep::Tube(rmid, rout, hthickl, 0.0, 2. * cms_units::piRadians);
      ns.addSolidNS(ns.prepend(name), solid);
      dd4hep::Volume glog2 = dd4hep::Volume(solid.name(), solid, matter1);
      ns.addVolumeNS(glog2);

#ifdef EDM_ML_DEBUG
      double eta1 = -log(tan(0.5 * atan(rmid / zz)));
      double eta2 = -log(tan(0.5 * atan(rout / zz)));
      edm::LogVerbatim("HGCalGeom") << name << " z|rin|rout " << zz << ":" << rmid << ":" << rout << " eta " << eta1
                                    << ":" << eta2;
      edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << solid.name() << " Tubs made of " << matter1.name()
                                    << " of dimensions " << rmid << ", " << rout << ", " << hthickl << ", 0.0, 360.0";
#endif
      zpos += hthickl;

      dd4hep::Position r1(0, 0, zpos);
      glog1.placeVolume(glog2, copy, r1);

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: Position " << glog2.name() << " number " << copy << " in "
                                    << glog1.name() << " at " << r1 << " with no rotation";
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

    solid = dd4hep::Tube(rin, rmid, hthick, 0.0, 2. * cms_units::piRadians);
    ns.addSolidNS(ns.prepend(name), solid);
    glog1 = dd4hep::Volume(solid.name(), solid, matter);
    ns.addVolumeNS(glog1);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << solid.name() << " Tubs made of " << matter.name()
                                  << " of dimensions " << rin << ", " << rmid << ", " << hthick << ", 0.0, 360.0";
#endif

    glog.placeVolume(glog1, 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << glog1.name() << " number 1 positioned in " << glog.name()
                                  << " at (0, 0, 0) with no rotation";
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
      edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: Layer " << ly << ":" << ii << " R " << rin << ":" << rmid
                                    << " Thick " << layerThickBot_[ii];
#endif

      dd4hep::Material matter1 = ns.material(materialsBot_[ii]);
      solid = dd4hep::Tube(rin, rmid, hthickl, 0.0, 2. * cms_units::piRadians);
      ns.addSolidNS(ns.prepend(name), solid);
      dd4hep::Volume glog2 = dd4hep::Volume(solid.name(), solid, matter1);
      ns.addVolumeNS(glog2);

#ifdef EDM_ML_DEBUG
      double eta1 = -log(tan(0.5 * atan(rin / zz)));
      double eta2 = -log(tan(0.5 * atan(rmid / zz)));
      edm::LogVerbatim("HGCalGeom") << name << " z|rin|rout " << zz << ":" << rin << ":" << rmid << " eta " << eta1
                                    << ":" << eta2;
      edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << solid.name() << " Tubs made of " << matter1.name()
                                    << " of dimensions " << rin << ", " << rmid << ", " << hthickl << ", 0.0, 360.0";
#endif
      zpos += hthickl;

      dd4hep::Position r1(0, 0, zpos);
      glog1.placeVolume(glog2, copy, r1);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: Position " << glog2.name() << " number " << copy << " in "
                                    << glog1.name() << " at " << r1 << " with no rotation";
#endif
      if (layerSenseBot_[ly] != 0) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: z " << (zz + zpos) << " Center " << copy << ":"
                                      << (copy - firstLayer_) << ":" << layerCenter_[copy - firstLayer_];
#endif
        positionSensitive(ctxt, e, glog2, rin, rmid, zz + zpos, layerSenseBot_[ly], layerCenter_[copy - firstLayer_]);
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

  void positionSensitive(cms::DDParsingContext& ctxt,
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
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << glog.name() << " rout " << rout << " N " << N
                                  << " for maximum u, v Offset; Shift " << xyoff.first << ":" << xyoff.second
                                  << " WaferSize " << (waferSize_ + waferSepar_);
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
            glog.placeVolume(ns.volume(waferNames_[type]), copy, tran);

#ifdef EDM_ML_DEBUG
            ++ntype[type];
            edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: " << glog.name() << " number " << copy << " positioned in "
                                          << glog.name() << " at " << tran << " with no rotation";
#endif
          }
        }
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalHEAlgo: Maximum # of u " << ium << ":" << iumAll << " # of v " << ivm
                                  << ":" << ivmAll << " and " << nin << ":" << kount << ":" << ntot << " wafers ("
                                  << ntype[0] << ":" << ntype[1] << ":" << ntype[2] << ":" << ntype[3] << ":"
                                  << ntype[4] << ":" << ntype[5] << ") for " << glog.name() << " R " << rin << ":"
                                  << rout;
#endif
  }

  //Required data members to cache the values from XML file
  HGCalGeomTools geomTools_;
  std::unique_ptr<HGCalWaferType> waferType_;

  std::vector<std::string> waferNames_;    // Wafer names
  std::vector<std::string> materials_;     // Materials
  std::vector<std::string> volumeNames_;   // Names
  std::vector<double> thickness_;          // Thickness of the material
  std::vector<int> copyNumber_;            // Initial copy numbers
  std::vector<int> layerNumbers_;          // Number of layers in a section
  std::vector<double> layerThick_;         // Thickness of each section
  std::vector<double> rMixLayer_;          // Partition between Si/Sci part
  std::vector<int> layerType_;             // Type of the layer
  std::vector<int> layerSense_;            // Content of a layer (sensitive?)
  int firstLayer_;                         // Copy # of the first sensitive layer
  int absorbMode_;                         // Absorber mode
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

  double zMinBlock_;                 // Starting z-value of the block
  std::vector<double> rad100to200_;  // Parameters for 120-200mum trans.
  std::vector<double> rad200to300_;  // Parameters for 200-300mum trans.
  double zMinRadPar_;                // Minimum z for radius parametriz.
  int choiceType_;                   // Type of parametrization to be used
  int nCutRadPar_;                   // Cut off threshold for corners
  double fracAreaMin_;               // Minimum fractional conatined area
  double waferSize_;                 // Width of the wafer
  double waferSepar_;                // Sensor separation
  int sectors_;                      // Sectors
  std::vector<double> slopeB_;       // Slope at the lower R
  std::vector<double> zFrontB_;      // Starting Z values for the slopes
  std::vector<double> rMinFront_;    // Corresponding rMin's
  std::vector<double> slopeT_;       // Slopes at the larger R
  std::vector<double> zFrontT_;      // Starting Z values for the slopes
  std::vector<double> rMaxFront_;    // Corresponding rMax's
  std::unordered_set<int> copies_;   // List of copy #'s
  double alpha_, cosAlpha_;

  static constexpr double tol1_ = 0.01;
  static constexpr double tol2_ = 0.00001;
};

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  HGCalHEAlgo healgo(ctxt, e);
  return cms::s_executed;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalHEAlgo, algorithm)
