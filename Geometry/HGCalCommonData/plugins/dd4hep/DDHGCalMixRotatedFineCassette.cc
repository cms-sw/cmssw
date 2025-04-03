///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalMixRotatedFineFineCassette.cc
// Description: Geometry factory class for HGCal (Mix)
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/Math/interface/angle_units.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DetectorDescription/DDCMS/interface/DDutils.h"
#include "DD4hep/DetFactoryHelper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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

struct HGCalMixRotatedFineCassette {
  HGCalMixRotatedFineCassette() {
    throw cms::Exception("HGCalGeom") << "Wrong initialization to HGCalMixRotatedFineCassette";
  }

  HGCalMixRotatedFineCassette(cms::DDParsingContext& ctxt, xml_h e) {
    cms::DDNamespace ns(ctxt, e, true);
    cms::DDAlgoArguments args(ctxt, e);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Creating an instance";
#endif
    static constexpr double tol1 = 0.01 * dd4hep::mm;

    dd4hep::Volume mother = ns.volume(args.parentName());

    waferTypes_ = args.value<int>("WaferTypes");
    passiveTypes_ = args.value<int>("PassiveTypes");
    facingTypes_ = args.value<int>("FacingTypes");
    orientationTypes_ = args.value<int>("OrientationTypes");
    partialTypes_ = args.value<int>("PartialTypes");
    placeOffset_ = args.value<int>("PlaceOffset");
    phiBinsScint_ = args.value<int>("NPhiBinScint");
    phiBinsFineScint_ = args.value<int>("NPhiBinFineScint");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette::Number of types of wafers: " << waferTypes_
                                  << " passives: " << passiveTypes_ << " facings: " << facingTypes_
                                  << " Orientations: " << orientationTypes_ << " PartialTypes: " << partialTypes_
                                  << " PlaceOffset: " << placeOffset_ << "; number of cells along phi "
                                  << phiBinsFineScint_ << ":" << phiBinsScint_;
#endif
    firstFineLayer_ = args.value<int>("FirstFineLayer");
    firstCoarseLayer_ = args.value<int>("FirstCoarseLayer");
    absorbMode_ = args.value<int>("AbsorberMode");
    sensitiveMode_ = args.value<int>("SensitiveMode");
    passiveMode_ = args.value<int>("PassiveMode");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette::First Layers " << firstFineLayer_ << ":"
                                  << firstCoarseLayer_ << " and Absober:Sensitive mode " << absorbMode_ << ":"
                                  << sensitiveMode_ << ":" << passiveMode_;
#endif
    zMinBlock_ = args.value<double>("zMinBlock");
    waferSize_ = args.value<double>("waferSize");
    waferSepar_ = args.value<double>("SensorSeparation");
    sectors_ = args.value<int>("Sectors");
    cassettes_ = args.value<int>("Cassettes");
    alpha_ = (1._pi) / sectors_;
    cosAlpha_ = cos(alpha_);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: zStart " << zMinBlock_ << " wafer width "
                                  << waferSize_ << " separations " << waferSepar_ << " sectors " << sectors_ << ":"
                                  << convertRadToDeg(alpha_) << ":" << cosAlpha_ << " with " << cassettes_
                                  << " cassettes";
#endif
    slopeB_ = args.value<std::vector<double>>("SlopeBottom");
    zFrontB_ = args.value<std::vector<double>>("ZFrontBottom");
    rMinFront_ = args.value<std::vector<double>>("RMinFront");
    slopeT_ = args.value<std::vector<double>>("SlopeTop");
    zFrontT_ = args.value<std::vector<double>>("ZFrontTop");
    rMaxFront_ = args.value<std::vector<double>>("RMaxFront");
#ifdef EDM_ML_DEBUG
    for (unsigned int i = 0; i < slopeB_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Bottom Block [" << i << "] Zmin " << zFrontB_[i] << " Rmin " << rMinFront_[i]
                                    << " Slope " << slopeB_[i];
    for (unsigned int i = 0; i < slopeT_.size(); ++i)
      edm::LogVerbatim("HGCalGeom") << "Top Block [" << i << "] Zmin " << zFrontT_[i] << " Rmax " << rMaxFront_[i]
                                    << " Slope " << slopeT_[i];
#endif
    waferFull_ = args.value<std::vector<std::string>>("WaferNamesFull");
    waferPart_ = args.value<std::vector<std::string>>("WaferNamesPartial");
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << waferFull_.size() << " full and "
                                  << waferPart_.size()
                                  << " partial modules\nDDHGCalMixRotatedFineCassette:Full Modules:";
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
    passiveFull_ = args.value<std::vector<std::string>>("PassiveNamesFull");
    passivePart_ = args.value<std::vector<std::string>>("PassiveNamesPartial");
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
    materials_ = args.value<std::vector<std::string>>("MaterialNames");
    names_ = args.value<std::vector<std::string>>("VolumeNames");
    thick_ = args.value<std::vector<double>>("Thickness");
    copyNumber_.resize(materials_.size(), 1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << materials_.size() << " types of volumes";
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
    layerOrient_ = args.value<std::vector<int>>("LayerTypes");
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
    materialTop_ = args.value<std::vector<std::string>>("TopMaterialNames");
    namesTop_ = args.value<std::vector<std::string>>("TopVolumeNames");
    layerThickTop_ = args.value<std::vector<double>>("TopLayerThickness");
    layerTypeTop_ = args.value<std::vector<int>>("TopLayerType");
    copyNumberTop_.resize(materialTop_.size(), firstFineLayer_);
    coverTypeTop_ = args.value<int>("TopCoverLayerType");
    coverTopLayers_ = args.value<int>("TopCoverLayers");
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
    tileFineRMin_ = args.value<std::vector<double>>("Tile6RMin");
    tileFineRMax_ = args.value<std::vector<double>>("Tile6RMax");
    tileFineIndex_ = args.value<std::vector<int>>("Tile6LayerRings");
    tileFinePhis_ = args.value<std::vector<int>>("Tile6PhiRange");
    tileFineLayerStart_ = args.value<std::vector<int>>("Tile6LayerStart");
    tileCoarseRMin_ = args.value<std::vector<double>>("TileRMin");
    tileCoarseRMax_ = args.value<std::vector<double>>("TileRMax");
    tileCoarseIndex_ = args.value<std::vector<int>>("TileLayerRings");
    tileCoarsePhis_ = args.value<std::vector<int>>("TilePhiRange");
    tileCoarseLayerStart_ = args.value<std::vector<int>>("TileLayerStart");
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
                                    << "Layer " << std::get<0>(HGCalTileIndex::tileUnpack(tileFineIndex_[k]))
                                    << " Ring " << std::get<1>(HGCalTileIndex::tileUnpack(tileFineIndex_[k])) << ":"
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
    std::vector<double> retract = args.value<std::vector<double>>("ScintRetract");
    double dphi = M_PI / cassettes_;
    for (int k = 0; k < cassettes_; ++k) {
      double phi = (2 * k + 1) * dphi;
      cassetteShiftScnt_.emplace_back(retract[k] * cos(phi));
      cassetteShiftScnt_.emplace_back(retract[k] * sin(phi));
    }
#ifdef EDM_ML_DEBUG
    unsigned int j2max = cassetteShiftScnt_.size();
    for (unsigned int j1 = 0; j1 < j2max; j1 += 6) {
      std::ostringstream st1;
      unsigned int j2 = std::min((j1 + 6), j2max);
      for (unsigned int j = j1; j < j2; ++j)
        st1 << " [" << j << "] " << std::setw(9) << cassetteShiftScnt_[j];
      edm::LogVerbatim("HGCalGeom") << st1.str();
    }
#endif
    cassette_.setParameter(cassettes_, cassetteShift_, false);
    cassette_.setParameterScint(cassetteShiftScnt_);

    ////////////////////////////////////////////////////////////////////
    // DDHGCalMixRotatedFineCassette methods...
    ////////////////////////////////////////////////////////////////////

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "==>> Constructing DDHGCalMixRotatedFineCassette...";
    copies_.clear();
#endif

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
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Layer " << ly << ":" << ii << " Front "
                                      << cms::convert2mm(zi) << ", " << cms::convert2mm(routF) << " Back "
                                      << cms::convert2mm(zo) << ", " << cms::convert2mm(rinB)
                                      << " superlayer thickness " << layerThick_[i];
#endif
        dd4hep::Material matter = ns.material(materials_[ii]);
        dd4hep::Volume glog;
        if (layerSense_[ly] <= 0) {
          std::vector<double> pgonZ, pgonRin, pgonRout;
          double rmax =
              (std::min(routF, HGCalGeomTools::radius(zz + hthick, zFrontT_, rMaxFront_, slopeT_)) * cosAlpha_) - tol1;
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
          edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << solid.name() << " polyhedra of "
                                        << sectors_ << " sectors covering " << convertRadToDeg(-alpha_) << ":"
                                        << convertRadToDeg(-alpha_ + 2._pi) << " with " << pgonZ.size() << " sections";
          for (unsigned int k = 0; k < pgonZ.size(); ++k)
            edm::LogVerbatim("HGCalGeom") << "[" << k << "] z " << cms::convert2mm(pgonZ[k]) << " R "
                                          << cms::convert2mm(pgonRin[k]) << ":" << cms::convert2mm(pgonRout[k]);
#endif
	  if (layerSense_[ly] < 0) {
	    int absType = -layerSense_[ly];
	    unsigned int num = (absType <= waferTypes_) ? passiveFull_.size() : passivePart_.size();
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "Abstype " << absType << " num " << num;
#endif
          if (num > 1)
            positionMix(ctxt, e, glog, name, copy, thick_[ii], matter, absType, fine);
	  }
	} else {
          double rins = (sensitiveMode_ < 1) ? rinB : HGCalGeomTools::radius(zz + hthick, zFrontB_, rMinFront_, slopeB_);
          double routs = (sensitiveMode_ < 1) ? routF : HGCalGeomTools::radius(zz - hthick, zFrontT_, rMaxFront_, slopeT_);
          dd4hep::Solid solid = dd4hep::Tube(rins, routs, hthick, 0.0, 2._pi);
          ns.addSolidNS(ns.prepend(name), solid);
          glog = dd4hep::Volume(solid.name(), solid, matter);
          ns.addVolumeNS(glog);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << solid.name() << " Tubs made of "
                                        << matter.name() << " of dimensions " << cms::convert2mm(rinB) << ":"
                                        << cms::convert2mm(rins) << ", " << cms::convert2mm(routF) << ":"
                                        << cms::convert2mm(routs) << ", " << cms::convert2mm(hthick)
                                        << ", 0.0, 360.0 and positioned in: " << glog.name() << " number " << copy;
#endif
          positionMix(ctxt, e, glog, name, copy, thick_[ii], matter, -layerSense_[ly], fine);
        }
        dd4hep::Position r1(0, 0, zz);
        mother.placeVolume(glog, copy, r1);
        ++copyNumber_[ii];
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << glog.name() << " number " << copy
                                      << " positioned in " << mother.name() << " at (0,0," << cms::convert2mm(zz)
                                      << ") with no rotation";
#endif
        zz += hthick;
      }  // End of loop over layers in a block
      zi = zo;
      laymin = laymax;
      // Make consistency check of all the partitions of the block
      if (std::abs(thickTot - layerThick_[i]) >= tol2_) {
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

  void positionMix(cms::DDParsingContext& ctxt,
                   xml_h e,
                   const dd4hep::Volume& glog,
                   const std::string& nameM,
                   int copyM,
                   double thick,
                   const dd4hep::Material& matter,
                   int absType,
                   bool fine) {
    cms::DDNamespace ns(ctxt, e, true);

    // Make the top part first
    for (unsigned int ly = 0; ly < layerTypeTop_.size(); ++ly) {
      int ii = layerTypeTop_[ly];
      copyNumberTop_[ii] = copyM;
    }
    double hthick = 0.5 * thick;
    double dphi = (fine) ? ((2._pi) / phiBinsFineScint_) : ((2._pi) / phiBinsScint_);
    double thickTot(0), zpos(-hthick);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Entry to positionMix with Name " << nameM
                                  << " copy " << copyM << " Thick " << thick << " AbsType " << absType << " Fine "
                                  << fine << " dphi " << convertRadToDeg(dphi);
#endif
    if (absType < 0) {
      for (unsigned int ly = 0; ly < layerTypeTop_.size(); ++ly) {
        int ii = layerTypeTop_[ly];
        int copy = copyNumberTop_[ii];
        int layer = (fine) ? (copy - firstFineLayer_) : (copy - firstCoarseLayer_);
        double hthickl = 0.5 * layerThickTop_[ii];
        thickTot += layerThickTop_[ii];
        zpos += hthickl;
        dd4hep::Material matter1 = ns.material(materialTop_[ii]);
        unsigned int k = 0;
        int firstTile = (fine) ? tileFineLayerStart_[layer] : tileCoarseLayerStart_[layer];
        int lastTile = (fine) ? ((layer + 1 < static_cast<int>(tileFineLayerStart_.size()))
                                     ? tileFineLayerStart_[layer + 1]
                                     : static_cast<int>(tileFineIndex_.size()))
                              : ((layer + 1 < static_cast<int>(tileCoarseLayerStart_.size()))
                                     ? tileCoarseLayerStart_[layer + 1]
                                     : static_cast<int>(tileCoarseIndex_.size()));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Layer " << ly << ":" << ii << ":" << layer
                                      << " Copy " << copy << " Tiles " << firstTile << ":" << lastTile << " Size "
                                      << tileFineIndex_.size() << ":" << tileCoarseIndex_.size() << " Fine " << fine
                                      << " absType " << absType;
#endif
        for (int ti = firstTile; ti < lastTile; ++ti) {
          double r1, r2;
          int cassette, fimin, fimax;
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: ti " << ti << ":" << fine << " index "
                                        << tileFineIndex_.size() << ":" << tileCoarseIndex_.size() << " Phis "
                                        << tileFinePhis_.size() << ":" << tileCoarsePhis_.size();
#endif
          if (fine) {
            r1 = tileFineRMin_[std::get<1>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti])) - 1];
            r2 = tileFineRMax_[std::get<2>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti])) - 1];
            cassette = std::get<0>(HGCalTileIndex::tileUnpack(tileFinePhis_[ti]));
            fimin = std::get<1>(HGCalTileIndex::tileUnpack(tileFinePhis_[ti]));
            fimax = std::get<2>(HGCalTileIndex::tileUnpack(tileFinePhis_[ti]));
          } else {
            r1 = tileCoarseRMin_[std::get<1>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti])) - 1];
            r2 = tileCoarseRMax_[std::get<2>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti])) - 1];
            cassette = std::get<0>(HGCalTileIndex::tileUnpack(tileCoarsePhis_[ti]));
            fimin = std::get<1>(HGCalTileIndex::tileUnpack(tileCoarsePhis_[ti]));
            fimax = std::get<2>(HGCalTileIndex::tileUnpack(tileCoarsePhis_[ti]));
          }
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Casstee|Fimin|Fimax " << cassette << ":"
                                        << fimin << ":" << fimax;
#endif
          double phi1 = dphi * (fimin - 1);
          double phi2 = dphi * (fimax - fimin + 1);
          auto cshift = cassette_.getShift(layer + 1, 1, cassette, true);
#ifdef EDM_ML_DEBUG
          int cassette0 = HGCalCassette::cassetteType(2, 1, cassette);  //
          int ir1 = (fine) ? std::get<1>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti]))
                           : std::get<1>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti]));
          int ir2 = (fine) ? std::get<2>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti]))
                           : std::get<2>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti]));
          edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Layer " << copy << ":" << (layer + 1)
                                        << " iR " << ir1 << ":" << ir2 << " R " << cms::convert2mm(r1) << ":"
                                        << cms::convert2mm(r2) << " Thick " << cms::convert2mm(2.0 * hthickl) << " phi "
                                        << fimin << ":" << fimax << ":" << convertRadToDeg(phi1) << ":"
                                        << convertRadToDeg(phi2) << " cassette " << cassette << ":" << cassette0
                                        << " Shift " << cms::convert2mm(cshift.first) << ":"
                                        << cms::convert2mm(cshift.second);
#endif
          std::string name = namesTop_[ii] + "L" + std::to_string(copy) + "F" + std::to_string(k);
          ++k;
          dd4hep::Solid solid = dd4hep::Tube(r1, r2, hthickl, phi1, phi2);
          ns.addSolidNS(ns.prepend(name), solid);
          dd4hep::Volume glog1 = dd4hep::Volume(solid.name(), solid, matter1);
          ns.addVolumeNS(glog1);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << glog1.name() << " Tubs made of "
                                        << matter1.name() << " of dimensions " << cms::convert2mm(r1) << ", "
                                        << cms::convert2mm(r2) << ", " << cms::convert2mm(hthickl) << ", "
                                        << convertRadToDeg(phi1) << ", " << convertRadToDeg(phi2);
#endif
          dd4hep::Position tran(-cshift.first, cshift.second, zpos);
          glog.placeVolume(glog1, copy, tran);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Position " << glog1.name() << " number "
                                        << copy << " in " << glog.name() << " at (" << cms::convert2mm(cshift.first)
                                        << ", " << cms::convert2mm(cshift.second) << ", " << cms::convert2mm(zpos)
                                        << ") with no rotation";
#endif
        }
        ++copyNumberTop_[ii];
        zpos += hthickl;
      }
      if (std::abs(thickTot - thick) >= tol2_) {
        if (thickTot > thick) {
          edm::LogError("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Thickness of the partition "
                                     << cms::convert2mm(thick) << " is smaller than " << cms::convert2mm(thickTot)
                                     << ": thickness of all its components in the top part **** ERROR ****";
        } else {
          edm::LogWarning("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Thickness of the partition "
                                       << cms::convert2mm(thick) << " does not match with " << cms::convert2mm(thickTot)
                                       << " of the components in top part";
        }
      }
    } else {
      int ii = coverTypeTop_;
      int copy = copyNumberCoverTop_[absType - 1];
      int layer = (fine) ? (copy - firstFineLayer_) : (copy - firstCoarseLayer_);
      double hthickl = 0.5 * layerThickTop_[ii];
      zpos += hthickl;
      dd4hep::Material matter1 = ns.material(materialTop_[ii]);
      unsigned int k = 0;
      int firstTile = (fine) ? tileFineLayerStart_[layer] : tileCoarseLayerStart_[layer];
      int lastTile = (fine) ? (((layer + 1) < static_cast<int>(tileFineLayerStart_.size()))
                                   ? tileFineLayerStart_[layer + 1]
                                   : static_cast<int>(tileFineIndex_.size()))
                            : (((layer + 1) < static_cast<int>(tileCoarseLayerStart_.size()))
                                   ? tileCoarseLayerStart_[layer + 1]
                                   : static_cast<int>(tileCoarseIndex_.size()));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: TOP Layer " << ii << ":" << layer << " Copy "
                                    << copy << " Tiles " << firstTile << ":" << lastTile << " Size "
                                    << tileFineIndex_.size() << ":" << tileCoarseIndex_.size() << " Fine " << fine
                                    << " absType " << absType;
#endif
      for (int ti = firstTile; ti < lastTile; ++ti) {
        double r1, r2;
        int cassette, fimin, fimax;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: ti " << ti << ":" << fine << " index "
                                      << tileFineIndex_.size() << ":" << tileCoarseIndex_.size() << " Phis "
                                      << tileFinePhis_.size() << ":" << tileCoarsePhis_.size();
#endif
        if (fine) {
          r1 = tileFineRMin_[std::get<1>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti])) - 1];
          r2 = tileFineRMax_[std::get<2>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti])) - 1];
          cassette = std::get<0>(HGCalTileIndex::tileUnpack(tileFinePhis_[ti]));
          fimin = std::get<1>(HGCalTileIndex::tileUnpack(tileFinePhis_[ti]));
          fimax = std::get<2>(HGCalTileIndex::tileUnpack(tileFinePhis_[ti]));
        } else {
          r1 = tileCoarseRMin_[std::get<1>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti])) - 1];
          r2 = tileCoarseRMax_[std::get<2>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti])) - 1];
          cassette = std::get<0>(HGCalTileIndex::tileUnpack(tileCoarsePhis_[ti]));
          fimin = std::get<1>(HGCalTileIndex::tileUnpack(tileCoarsePhis_[ti]));
          fimax = std::get<2>(HGCalTileIndex::tileUnpack(tileCoarsePhis_[ti]));
        }
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Casstee|Fimin|Fimax " << cassette << ":"
                                      << fimin << ":" << fimax;
#endif
        double phi1 = dphi * (fimin - 1);
        double phi2 = dphi * (fimax - fimin + 1);
        auto cshift = cassette_.getShift(layer + 1, 1, cassette, true);
#ifdef EDM_ML_DEBUG
        int cassette0 = HGCalCassette::cassetteType(2, 1, cassette);  //
        int ir1 = (fine) ? std::get<1>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti]))
                         : std::get<1>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti]));
        int ir2 = (fine) ? std::get<2>(HGCalTileIndex::tileUnpack(tileFineIndex_[ti]))
                         : std::get<2>(HGCalTileIndex::tileUnpack(tileCoarseIndex_[ti]));
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Layer " << copy << ":" << (layer + 1) << " iR "
                                      << ir1 << ":" << ir2 << " R " << cms::convert2mm(r1) << ":" << cms::convert2mm(r2)
                                      << " Thick " << cms::convert2mm(2.0 * hthickl) << " phi " << fimin << ":" << fimax
                                      << ":" << convertRadToDeg(phi1) << ":" << convertRadToDeg(phi2) << " cassette "
                                      << cassette << ":" << cassette0 << " Shift " << cms::convert2mm(cshift.first)
                                      << ":" << cms::convert2mm(cshift.second);
#endif
        std::string name = namesTop_[ii] + "L" + std::to_string(copy) + "F" + std::to_string(k);
        ++k;
        dd4hep::Solid solid = dd4hep::Tube(r1, r2, hthickl, phi1, phi2);
        ns.addSolidNS(ns.prepend(name), solid);
        dd4hep::Volume glog1 = dd4hep::Volume(solid.name(), solid, matter1);
        ns.addVolumeNS(glog1);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << glog1.name() << " Tubs made of "
                                      << matter1.name() << " of dimensions " << cms::convert2mm(r1) << ", "
                                      << cms::convert2mm(r2) << ", " << cms::convert2mm(hthickl) << ", "
                                      << convertRadToDeg(phi1) << ", " << convertRadToDeg(phi2);
#endif
        dd4hep::Position tran(-cshift.first, cshift.second, zpos);
        glog.placeVolume(glog1, copy, tran);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Position " << glog1.name() << " number "
                                      << copy << " in " << glog.name() << " at (" << cms::convert2mm(-cshift.first)
                                      << ", " << cms::convert2mm(cshift.second) << ", " << cms::convert2mm(zpos)
                                      << ") with no rotation";
#endif
      }
      ++copyNumberCoverTop_[absType - 1];
    }

    // Make the bottom part next
    int layer = (copyM - firstFineLayer_);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Start bottom section for layer " << layer
                                  << " absType " << absType;
#endif
    if (absType > 0) {
#ifdef EDM_ML_DEBUG
      int kount(0);
#endif
      for (int k = 0; k < cassettes_; ++k) {
        int cassette = k + 1;
        auto cshift = cassette_.getShift(layer + 1, -1, cassette);
        double xpos = -cshift.first;
        double ypos = cshift.second;
        int i = layer * cassettes_ + k;
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette::Passive: layer " << layer + 1 << " cassette "
                                      << cassette << " Shift " << cms::convert2mm(-cshift.first) << ":"
                                      << cms::convert2mm(cshift.second) << " PassiveIndex " << i << ":"
                                      << passiveFull_.size() << ":" << passivePart_.size();
#endif
        std::string passive = (absType <= waferTypes_) ? passiveFull_[i] : passivePart_[i];
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Passive " << passive << " number " << cassette
                                      << " pos " << cms::convert2mm(xpos) << ":" << cms::convert2mm(ypos);
        kount++;
#endif
        dd4hep::Position tran(xpos, ypos, 0.0);
        glog.placeVolume(ns.volume(passive), cassette, tran);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << passive << " number " << cassette
                                      << " positioned in " << glog.name() << " at (" << cms::convert2mm(xpos) << ","
                                      << cms::convert2mm(ypos) << ",0) with no rotation";
#endif
      }
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: " << kount << " passives of type " << absType
                                    << " for " << glog.name();
#endif
    } else {
      static const double sqrt3 = std::sqrt(3.0);
      int layercenter = layerOrient_[layer];
      int layertype = HGCalTypes::layerFrontBack(layerOrient_[layer]);
      int firstWafer = waferLayerStart_[layer];
      int lastWafer =
          (((layer + 1) < static_cast<int>(waferLayerStart_.size())) ? waferLayerStart_[layer + 1]
                                                                     : static_cast<int>(waferIndex_.size()));
      double delx = 0.5 * (waferSize_ + waferSepar_);
      double dely = 2.0 * delx / sqrt3;
      double dy = 0.75 * dely;
      const auto& xyoff = geomTools_.shiftXY(layercenter, (waferSize_ + waferSepar_));
#ifdef EDM_ML_DEBUG
      int ium(0), ivm(0), kount(0);
      std::vector<int> ntype(3, 0);
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette::Bottom: " << glog.name() << "  r "
                                    << cms::convert2mm(delx) << " R " << cms::convert2mm(dely) << " dy "
                                    << cms::convert2mm(dy) << " Shift " << cms::convert2mm(xyoff.first) << ":"
                                    << cms::convert2mm(xyoff.second) << " WaferSize "
                                    << cms::convert2mm(waferSize_ + waferSepar_) << " index " << firstWafer << ":"
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
            << "DDHGCalMixRotatedFineCassette::index:Property:layertype:type:part:orien:cassette:place:offsets:ind "
            << k << waferProperty_[k] << ":" << layertype << ":" << type << ":" << part << ":" << orien << ":"
            << cassette << ":" << place;
#endif
        auto cshift = cassette_.getShift(layer + 1, -1, cassette, false);
        double xpos = xyoff.first - cshift.first + nc * delx;
        double ypos = xyoff.second + cshift.second + nr * dy;
#ifdef EDM_ML_DEBUG
        double xorig = xyoff.first + nc * delx;
        double yorig = xyoff.second + nr * dy;
        double angle = std::atan2(yorig, xorig);
        edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette::Wafer: layer " << layer + 1 << " cassette "
                                      << cassette << " Shift " << cms::convert2mm(cshift.first) << ":"
                                      << cms::convert2mm(cshift.second) << " Original " << cms::convert2mm(xorig) << ":"
                                      << cms::convert2mm(yorig) << ":" << convertRadToDeg(angle) << " Final "
                                      << cms::convert2mm(xpos) << ":" << cms::convert2mm(ypos);
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
        dd4hep::Position tran(xpos, ypos, 0.0);
        glog.placeVolume(ns.volume(wafer), copy, tran);
#ifdef EDM_ML_DEBUG
        ++ntype[type];
        edm::LogVerbatim("HGCalGeom") << " DDHGCalMixRotatedFineCassette: " << wafer << " number " << copy << " type "
                                      << layertype << ":" << type << " positioned in " << glog.name() << " at ("
                                      << cms::convert2mm(xpos) << "," << cms::convert2mm(ypos)
                                      << ",0) with no rotation";
#endif
      }
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalMixRotatedFineCassette: Maximum # of u " << ium << " # of v " << ivm
                                    << " and " << kount << " wafers (" << ntype[0] << ":" << ntype[1] << ":" << ntype[2]
                                    << ") for " << glog.name();
#endif
    }
  }

  HGCalGeomTools geomTools_;
  HGCalCassette cassette_;

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
  std::vector<double> cassetteShiftScnt_;  // Shifts of the cassetes for scintillators
  std::string nameSpace_;                  // Namespace of this and ALL sub-parts
  std::unordered_set<int> copies_;         // List of copy #'s
  double alpha_, cosAlpha_;
  static constexpr double tol2_ = 0.00001 * dd4hep::mm;
};

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  HGCalMixRotatedFineCassette siliconRotatedCassetteAlgo(ctxt, e);
  return cms::s_executed;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalMixRotatedFineCassette, algorithm)
