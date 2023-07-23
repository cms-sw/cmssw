// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// Alignment
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/TrackerAlignment/interface/TrackerScenarioBuilder.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

// C++
#include <memory>
#include <algorithm>

///
/// An ESProducer that fills the TrackerDigiGeometryRcd with a misaligned tracker
///
/// This should replace the standard TrackerDigiGeometryESModule when producing
/// Misalignment scenarios.
///

class MisalignedTrackerESProducer : public edm::ESProducer {
public:
  /// Constructor
  MisalignedTrackerESProducer(const edm::ParameterSet& p);

  /// Destructor
  ~MisalignedTrackerESProducer() override;

  /// Produce the misaligned tracker geometry and store it
  std::unique_ptr<TrackerGeometry> produce(const TrackerDigiGeometryRecord& iRecord);

private:
  edm::ESGetToken<GeometricDet, IdealGeometryRecord> geomDetToken_;
  edm::ESGetToken<PTrackerParameters, PTrackerParametersRcd> ptpToken_;
  edm::ESGetToken<PTrackerAdditionalParametersPerDet, PTrackerAdditionalParametersPerDetRcd> ptitpToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

  const bool theSaveToDB;  /// whether or not writing to DB
  const bool
      theSaveFakeScenario;  /// if theSaveToDB is true, save a fake scenario (empty alignments), irrespective of the misalignment scenario below
  const edm::ParameterSet theScenario;  /// misalignment scenario
  const std::string theAlignRecordName, theErrorRecordName;
};

//__________________________________________________________________________________________________
//__________________________________________________________________________________________________
//__________________________________________________________________________________________________

//__________________________________________________________________________________________________
MisalignedTrackerESProducer::MisalignedTrackerESProducer(const edm::ParameterSet& p)
    : theSaveToDB(p.getUntrackedParameter<bool>("saveToDbase")),
      theSaveFakeScenario(p.getUntrackedParameter<bool>("saveFakeScenario")),
      theScenario(p.getParameter<edm::ParameterSet>("scenario")),
      theAlignRecordName("TrackerAlignmentRcd"),
      theErrorRecordName("TrackerAlignmentErrorExtendedRcd") {
  auto cc = setWhatProduced(this);
  geomDetToken_ = cc.consumes();
  ptpToken_ = cc.consumes();
  ptitpToken_ = cc.consumes();
  topoToken_ = cc.consumes();
}

//__________________________________________________________________________________________________
MisalignedTrackerESProducer::~MisalignedTrackerESProducer() {}

//__________________________________________________________________________________________________
std::unique_ptr<TrackerGeometry> MisalignedTrackerESProducer::produce(const TrackerDigiGeometryRecord& iRecord) {
  //Retrieve tracker topology from geometry
  const TrackerTopology* tTopo = &iRecord.get(topoToken_);

  edm::LogInfo("MisalignedTracker") << "Producer called";

  // Create the tracker geometry from ideal geometry
  const GeometricDet* gD = &iRecord.get(geomDetToken_);
  const PTrackerParameters& ptp = iRecord.get(ptpToken_);
  const PTrackerAdditionalParametersPerDet* ptitp = &iRecord.get(ptitpToken_);

  TrackerGeomBuilderFromGeometricDet trackerBuilder;
  std::unique_ptr<TrackerGeometry> theTracker(trackerBuilder.build(gD, ptitp, ptp, tTopo));

  // Create the alignable hierarchy
  auto theAlignableTracker = std::make_unique<AlignableTracker>(&(*theTracker), tTopo);

  // Create misalignment scenario, apply to geometry
  TrackerScenarioBuilder scenarioBuilder(&(*theAlignableTracker));
  scenarioBuilder.applyScenario(theScenario);
  Alignments alignments = *(theAlignableTracker->alignments());
  AlignmentErrorsExtended alignmentErrors = *(theAlignableTracker->alignmentErrors());

  // Store result to EventSetup
  GeometryAligner aligner;
  aligner.applyAlignments<TrackerGeometry>(&(*theTracker),
                                           &alignments,
                                           &alignmentErrors,
                                           AlignTransform());  // dummy global position

  // Write alignments to DB: have to sort beforhand!
  if (theSaveToDB) {
    // Call service
    edm::Service<cond::service::PoolDBOutputService> poolDbService;
    if (!poolDbService.isAvailable())  // Die if not available
      throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
    if (theSaveFakeScenario) {  // make empty!
      alignments.clear();
      alignmentErrors.clear();
    }
    poolDbService->writeOneIOV<Alignments>(alignments, poolDbService->currentTime(), theAlignRecordName);
    poolDbService->writeOneIOV<AlignmentErrorsExtended>(
        alignmentErrors, poolDbService->currentTime(), theErrorRecordName);
  }

  edm::LogInfo("MisalignedTracker") << "Producer done";
  return theTracker;
}

DEFINE_FWK_EVENTSETUP_MODULE(MisalignedTrackerESProducer);
