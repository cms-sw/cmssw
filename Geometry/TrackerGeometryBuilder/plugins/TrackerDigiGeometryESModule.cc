#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include <memory>
#include <string>

class TrackerDigiGeometryESModule : public edm::ESProducer {
public:
  TrackerDigiGeometryESModule(const edm::ParameterSet& p);
  ~TrackerDigiGeometryESModule() override;
  std::unique_ptr<TrackerGeometry> produce(const TrackerDigiGeometryRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  /// Called when geometry description changes
  const std::string alignmentsLabel_;
  const std::string myLabel_;

  edm::ESGetToken<GeometricDet, IdealGeometryRecord> geometricDetToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
  edm::ESGetToken<PTrackerParameters, PTrackerParametersRcd> trackerParamsToken_;

  edm::ESGetToken<Alignments, GlobalPositionRcd> globalAlignmentToken_;
  edm::ESGetToken<Alignments, TrackerAlignmentRcd> trackerAlignmentToken_;
  edm::ESGetToken<AlignmentErrorsExtended, TrackerAlignmentErrorExtendedRcd> alignmentErrorsToken_;
  edm::ESGetToken<AlignmentSurfaceDeformations, TrackerSurfaceDeformationRcd> deformationsToken_;

  const bool applyAlignment_;  // Switch to apply alignment corrections
};

//__________________________________________________________________
TrackerDigiGeometryESModule::TrackerDigiGeometryESModule(const edm::ParameterSet& p)
    : alignmentsLabel_(p.getParameter<std::string>("alignmentsLabel")),
      myLabel_(p.getParameter<std::string>("appendToDataLabel")),
      applyAlignment_(p.getParameter<bool>("applyAlignment")) {
  {
    auto cc = setWhatProduced(this);
    const edm::ESInputTag kEmptyTag;
    geometricDetToken_ = cc.consumesFrom<GeometricDet, IdealGeometryRecord>(kEmptyTag);
    trackerTopoToken_ = cc.consumesFrom<TrackerTopology, TrackerTopologyRcd>(kEmptyTag);
    trackerParamsToken_ = cc.consumesFrom<PTrackerParameters, PTrackerParametersRcd>(kEmptyTag);

    if (applyAlignment_) {
      const edm::ESInputTag kAlignTag{"", alignmentsLabel_};
      globalAlignmentToken_ = cc.consumesFrom<Alignments, GlobalPositionRcd>(kAlignTag);
      trackerAlignmentToken_ = cc.consumesFrom<Alignments, TrackerAlignmentRcd>(kAlignTag);
      alignmentErrorsToken_ = cc.consumesFrom<AlignmentErrorsExtended, TrackerAlignmentErrorExtendedRcd>(kAlignTag);
      deformationsToken_ = cc.consumesFrom<AlignmentSurfaceDeformations, TrackerSurfaceDeformationRcd>(kAlignTag);
    }
  }

  edm::LogInfo("Geometry") << "@SUB=TrackerDigiGeometryESModule"
                           << "Label '" << myLabel_ << "' " << (applyAlignment_ ? "looking for" : "IGNORING")
                           << " alignment labels '" << alignmentsLabel_ << "'.";
}

//__________________________________________________________________
TrackerDigiGeometryESModule::~TrackerDigiGeometryESModule() {}

void TrackerDigiGeometryESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription descDB;
  descDB.add<std::string>("appendToDataLabel", "");
  descDB.add<bool>("fromDDD", false);
  descDB.add<bool>("applyAlignment", true);
  descDB.add<std::string>("alignmentsLabel", "");
  descriptions.add("trackerGeometryDB", descDB);

  edm::ParameterSetDescription desc;
  desc.add<std::string>("appendToDataLabel", "");
  desc.add<bool>("fromDDD", true);
  desc.add<bool>("applyAlignment", true);
  desc.add<std::string>("alignmentsLabel", "");
  descriptions.add("trackerGeometry", desc);
}

//__________________________________________________________________
std::unique_ptr<TrackerGeometry> TrackerDigiGeometryESModule::produce(const TrackerDigiGeometryRecord& iRecord) {
  //
  // Called whenever the alignments, alignment errors or global positions change
  //
  auto const& gD = iRecord.get(geometricDetToken_);

  auto const& tTopo = iRecord.get(trackerTopoToken_);

  auto const& ptp = iRecord.get(trackerParamsToken_);

  TrackerGeomBuilderFromGeometricDet builder;
  std::unique_ptr<TrackerGeometry> tracker(builder.build(&gD, ptp, &tTopo));

  if (applyAlignment_) {
    // Since fake is fully working when checking for 'empty', we should get rid of applyAlignment_!
    auto const& globalPosition = iRecord.get(globalAlignmentToken_);
    auto const& alignments = iRecord.get(trackerAlignmentToken_);
    auto const& alignmentErrors = iRecord.get(alignmentErrorsToken_);
    // apply if not empty:
    if (alignments.empty() && alignmentErrors.empty() && globalPosition.empty()) {
      edm::LogInfo("Config") << "@SUB=TrackerDigiGeometryRecord::produce"
                             << "Alignment(Error)s and global position (label '" << alignmentsLabel_
                             << "') empty: Geometry producer (label "
                             << "'" << myLabel_ << "') assumes fake and does not apply.";
    } else {
      GeometryAligner ali;
      ali.applyAlignments<TrackerGeometry>(tracker.get(),
                                           &(alignments),
                                           &(alignmentErrors),
                                           align::DetectorGlobalPosition(globalPosition, DetId(DetId::Tracker)));
    }

    auto const& surfaceDeformations = iRecord.get(deformationsToken_);
    // apply if not empty:
    if (surfaceDeformations.empty()) {
      edm::LogInfo("Config") << "@SUB=TrackerDigiGeometryRecord::produce"
                             << "AlignmentSurfaceDeformations (label '" << alignmentsLabel_
                             << "') empty: Geometry producer (label "
                             << "'" << myLabel_ << "') assumes fake and does not apply.";
    } else {
      GeometryAligner ali;
      ali.attachSurfaceDeformations<TrackerGeometry>(tracker.get(), &(surfaceDeformations));
    }
  }

  return tracker;
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerDigiGeometryESModule);
