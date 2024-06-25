// -*- C++ -*-
//
// Package:    Alignment/TrackerAlignment
// Class:      MCMisalignmentScaler
//
/**\class MCMisalignmentScaler MCMisalignmentScaler.cc Alignment/TrackerAlignment/plugins/MCMisalignmentScaler.cc

   Description: Plugin to rescale misalignment wrt. ideal geometry

   Implementation:

   The plugin takes the ideal geometry and the alignment object and rescales
   the position difference by the scaling factor provided by the user.

*/
//
// Original Author:  Gregor Mittag
//         Created:  Tue, 10 Oct 2017 15:49:17 GMT
//
//

// system include files
#include <memory>
#include <unordered_map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "CondFormats/GeometryObjects/interface/PTrackerAdditionalParametersPerDet.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "Geometry/Records/interface/PTrackerAdditionalParametersPerDetRcd.h"

//
// class declaration
//

class MCMisalignmentScaler : public edm::one::EDAnalyzer<> {
public:
  explicit MCMisalignmentScaler(const edm::ParameterSet&);
  ~MCMisalignmentScaler() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::ESGetToken<SiPixelQuality, SiPixelQualityRcd> pixelQualityToken_;
  const edm::ESGetToken<SiStripQuality, SiStripQualityRcd> stripQualityToken_;
  const edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomDetToken_;
  const edm::ESGetToken<PTrackerParameters, PTrackerParametersRcd> ptpToken_;
  const edm::ESGetToken<PTrackerAdditionalParametersPerDet, PTrackerAdditionalParametersPerDetRcd> ptitpToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::ESGetToken<Alignments, TrackerAlignmentRcd> aliToken_;
  using ScalerMap = std::unordered_map<unsigned int, std::unordered_map<int, double> >;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  ScalerMap decodeSubDetectors(const edm::VParameterSet&);

  // ----------member data ---------------------------
  const ScalerMap scalers_;
  const bool pullBadModulesToIdeal_;
  const double outlierPullToIdealCut_;
  bool firstEvent_{true};
};

//
// constructors and destructor
//
MCMisalignmentScaler::MCMisalignmentScaler(const edm::ParameterSet& iConfig)
    : pixelQualityToken_(esConsumes()),
      stripQualityToken_(esConsumes()),
      geomDetToken_(esConsumes()),
      ptpToken_(esConsumes()),
      ptitpToken_(esConsumes()),
      topoToken_(esConsumes()),
      aliToken_(esConsumes()),
      scalers_{decodeSubDetectors(iConfig.getParameter<edm::VParameterSet>("scalers"))},
      pullBadModulesToIdeal_{iConfig.getUntrackedParameter<bool>("pullBadModulesToIdeal")},
      outlierPullToIdealCut_{iConfig.getUntrackedParameter<double>("outlierPullToIdealCut")} {}

//
// member functions
//

// ------------ method called for each event  ------------
void MCMisalignmentScaler::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  if (!firstEvent_)
    return;
  firstEvent_ = false;

  // get handle on bad modules
  const SiPixelQuality* pixelModules = &iSetup.getData(pixelQualityToken_);
  const SiStripQuality* stripModules = &iSetup.getData(stripQualityToken_);

  // get the tracker geometry
  const GeometricDet* geometricDet = &iSetup.getData(geomDetToken_);
  const PTrackerParameters& ptp = iSetup.getData(ptpToken_);
  const PTrackerAdditionalParametersPerDet* ptitp = &iSetup.getData(ptitpToken_);
  const TrackerTopology* topology = &iSetup.getData(topoToken_);

  TrackerGeomBuilderFromGeometricDet trackerBuilder;
  auto tracker = std::unique_ptr<TrackerGeometry>{trackerBuilder.build(geometricDet, ptitp, ptp, topology)};

  auto dets = tracker->dets();
  std::sort(dets.begin(), dets.end(), [](const auto& a, const auto& b) {
    return a->geographicalId().rawId() < b->geographicalId().rawId();
  });

  // get the input alignment
  const Alignments* alignments = &iSetup.getData(aliToken_);

  if (dets.size() != alignments->m_align.size()) {
    throw cms::Exception("GeometryMismatch") << "Size mismatch between alignments (size=" << alignments->m_align.size()
                                             << ") and ideal geometry (size=" << dets.size() << ")";
  }

  Alignments rescaledAlignments{};
  {
    auto outlierCounter{0};
    auto ideal = dets.cbegin();
    const auto& ideal_end = dets.cend();
    auto misaligned = alignments->m_align.cbegin();
    for (; ideal != ideal_end; ++ideal, ++misaligned) {
      if ((*ideal)->geographicalId().rawId() != misaligned->rawId()) {
        throw cms::Exception("GeometryMismatch") << "Order differs between Dets in alignments ideal geometry.";
      }

      // determine scale factor
      const auto& subDetId = (*ideal)->geographicalId().subdetId();
      auto side = topology->side((*ideal)->geographicalId());
      if (side == 0) {
        switch (subDetId) {
          case PixelSubdetector::PixelBarrel:
            side = 1;  // both sides are treated identical -> pick one of them
            break;
          case StripSubdetector::TIB:
            side = topology->tibSide((*ideal)->geographicalId());
            break;
          case StripSubdetector::TOB:
            side = topology->tobSide((*ideal)->geographicalId());
            break;
          default:
            break;
        }
      }
      auto scaleFactor = scalers_.find(subDetId)->second.find(side)->second;

      if (pullBadModulesToIdeal_ &&
          (pixelModules->IsModuleBad(misaligned->rawId()) || stripModules->IsModuleBad(misaligned->rawId()))) {
        scaleFactor = 0.0;
      }

      auto x_diff = misaligned->translation().x() - (*ideal)->position().x();
      auto y_diff = misaligned->translation().y() - (*ideal)->position().y();
      auto z_diff = misaligned->translation().z() - (*ideal)->position().z();

      auto xx_diff = misaligned->rotation().xx() - (*ideal)->rotation().xx();
      auto xy_diff = misaligned->rotation().xy() - (*ideal)->rotation().xy();
      auto xz_diff = misaligned->rotation().xz() - (*ideal)->rotation().xz();
      auto yx_diff = misaligned->rotation().yx() - (*ideal)->rotation().yx();
      auto yy_diff = misaligned->rotation().yy() - (*ideal)->rotation().yy();
      auto yz_diff = misaligned->rotation().yz() - (*ideal)->rotation().yz();
      auto zx_diff = misaligned->rotation().zx() - (*ideal)->rotation().zx();
      auto zy_diff = misaligned->rotation().zy() - (*ideal)->rotation().zy();
      auto zz_diff = misaligned->rotation().zz() - (*ideal)->rotation().zz();

      if (outlierPullToIdealCut_ > 0.0 &&
          (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff) > outlierPullToIdealCut_ * outlierPullToIdealCut_) {
        ++outlierCounter;
        edm::LogInfo("Alignment") << outlierCounter << ") Outlier found in subdetector " << subDetId
                                  << ":  delta x: " << x_diff << ",  delta y: " << y_diff << ",  delta z: " << z_diff
                                  << ",  delta xx: " << xx_diff << ",  delta xy: " << xy_diff
                                  << ",  delta xz: " << xz_diff << ",  delta yx: " << yx_diff
                                  << ",  delta yx: " << yy_diff << ",  delta yy: " << yz_diff
                                  << ",  delta zz: " << zx_diff << ",  delta zy: " << zy_diff
                                  << ",  delta zz: " << zz_diff << "\n";
        scaleFactor = 0.0;
      }

      const AlignTransform::Translation rescaledTranslation{(*ideal)->position().x() + scaleFactor * x_diff,
                                                            (*ideal)->position().y() + scaleFactor * y_diff,
                                                            (*ideal)->position().z() + scaleFactor * z_diff};

      const AlignTransform::Rotation rescaledRotation{
          CLHEP::HepRep3x3{(*ideal)->rotation().xx() + scaleFactor * xx_diff,
                           (*ideal)->rotation().xy() + scaleFactor * xy_diff,
                           (*ideal)->rotation().xz() + scaleFactor * xz_diff,
                           (*ideal)->rotation().yx() + scaleFactor * yx_diff,
                           (*ideal)->rotation().yy() + scaleFactor * yy_diff,
                           (*ideal)->rotation().yz() + scaleFactor * yz_diff,
                           (*ideal)->rotation().zx() + scaleFactor * zx_diff,
                           (*ideal)->rotation().zy() + scaleFactor * zy_diff,
                           (*ideal)->rotation().zz() + scaleFactor * zz_diff}};

      const AlignTransform rescaledTransform{rescaledTranslation, rescaledRotation, misaligned->rawId()};
      rescaledAlignments.m_align.emplace_back(rescaledTransform);
    }
  }

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (!poolDb.isAvailable()) {
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
  }
  edm::LogInfo("Alignment") << "Writing rescaled tracker-alignment record.";
  const auto& since = cond::timeTypeSpecs[cond::runnumber].beginValue;
  poolDb->writeOneIOV(rescaledAlignments, since, "TrackerAlignmentRcd");
}

MCMisalignmentScaler::ScalerMap MCMisalignmentScaler::decodeSubDetectors(const edm::VParameterSet& psets) {
  // initialize scaler map
  ScalerMap subDetMap;
  for (unsigned int subDetId = 1; subDetId <= 6; ++subDetId) {
    subDetMap[subDetId][1] = 1.0;
    subDetMap[subDetId][2] = 1.0;
  }

  // apply scale factors from configuration
  for (const auto& pset : psets) {
    const auto& name = pset.getUntrackedParameter<std::string>("subDetector");
    const auto& factor = pset.getUntrackedParameter<double>("factor");

    std::vector<int> sides;
    if (name.find('-') != std::string::npos)
      sides.push_back(1);
    if (name.find('+') != std::string::npos)
      sides.push_back(2);
    if (sides.empty()) {  // -> use both sides
      sides.push_back(1);
      sides.push_back(2);
    }

    if (name.find("Tracker") != std::string::npos) {
      for (unsigned int subDetId = 1; subDetId <= 6; ++subDetId) {
        for (const auto& side : sides)
          subDetMap[subDetId][side] *= factor;
      }
      if (sides.size() == 1) {
        // if only one side to be scaled
        // -> scale also the other side for PXB (subdetid = 1)
        subDetMap[PixelSubdetector::PixelBarrel][std::abs(sides[0] - 2) + 1] *= factor;
      }
    } else if (name.find("PXB") != std::string::npos) {
      // ignore sides for PXB
      subDetMap[PixelSubdetector::PixelBarrel][1] *= factor;
      subDetMap[PixelSubdetector::PixelBarrel][2] *= factor;
    } else if (name.find("PXF") != std::string::npos) {
      for (const auto& side : sides)
        subDetMap[PixelSubdetector::PixelEndcap][side] *= factor;
    } else if (name.find("TIB") != std::string::npos) {
      for (const auto& side : sides)
        subDetMap[StripSubdetector::TIB][side] *= factor;
    } else if (name.find("TOB") != std::string::npos) {
      for (const auto& side : sides)
        subDetMap[StripSubdetector::TOB][side] *= factor;
    } else if (name.find("TID") != std::string::npos) {
      for (const auto& side : sides)
        subDetMap[StripSubdetector::TID][side] *= factor;
    } else if (name.find("TEC") != std::string::npos) {
      for (const auto& side : sides)
        subDetMap[StripSubdetector::TEC][side] *= factor;
    } else {
      throw cms::Exception("BadConfig") << "@SUB=MCMisalignmentScaler::decodeSubDetectors\n"
                                        << "Unknown tracker subdetector: " << name
                                        << "\nSupported options: Tracker, PXB, PXF, TIB, TOB, TID, TEC "
                                        << "(possibly decorated with '+' or '-')";
    }
  }

  std::stringstream logInfo;
  logInfo << "MC misalignment scale factors:\n";
  for (const auto& subdet : subDetMap) {
    logInfo << "  Subdet " << subdet.first << "\n";
    for (const auto& side : subdet.second) {
      logInfo << "    side " << side.first << ": " << side.second << "\n";
    }
    logInfo << "\n";
  }
  edm::LogInfo("Alignment") << logInfo.str();

  return subDetMap;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void MCMisalignmentScaler::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment(
      "Creates rescaled MC misalignment scenario. "
      "PoolDBOutputService must be set up for 'TrackerAlignmentRcd'.");
  edm::ParameterSetDescription descScaler;
  descScaler.setComment(
      "ParameterSet specifying the tracker part to be scaled "
      "by a given factor.");
  descScaler.addUntracked<std::string>("subDetector", "Tracker");
  descScaler.addUntracked<double>("factor", 1.0);
  desc.addVPSet("scalers", descScaler, std::vector<edm::ParameterSet>(1));
  desc.addUntracked<bool>("pullBadModulesToIdeal", false);
  desc.addUntracked<double>("outlierPullToIdealCut", -1.0);
  descriptions.add("mcMisalignmentScaler", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MCMisalignmentScaler);
