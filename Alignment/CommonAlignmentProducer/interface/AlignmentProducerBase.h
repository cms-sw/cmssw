#ifndef Alignment_CommonAlignmentProducer_AlignmentProducerBase_h
#define Alignment_CommonAlignmentProducer_AlignmentProducerBase_h

/**
 * @package   Alignment/CommonAlignmentProducer
 * @file      AlignmentProducerBase.h
 *
 * @author    Gregor Mittag
 * @date      2017/03/16
 *
 * @brief     Alignment producer base class
 *
 * Abstract base class providing the functionality to produce alignments.
 * Derived classes can use its methods to implement the methods of the
 * respective Framework module.
 *
 * At the time of writing, this class can only be used within edm::one or legacy
 * modules because it uses the TFileService within the alignment monitors,
 * i.e. edm::one modules need to declare this resource in the constructor:
 *
 * 'usesResource(TFileService::kSharedResource);'.
 *
 */

#include <memory>

#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyErrorExtendedRcd.h"
#include "CondFormats/Common/interface/Time.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

class AlignTransform;
class Alignments;
class AlignmentErrorsExtended;
class AlignmentSurfaceDeformations;
class MuonGeometryRecord;
struct SurveyErrors;
class TrackerTopology;
class TrackerDigiGeometryRecord;

class AlignmentProducerBase {
protected:
  AlignmentProducerBase(const edm::ParameterSet&);

  // 'noexcept(false)' is needed currently for multiple inheritance with Framework modules
  virtual ~AlignmentProducerBase() noexcept(false);

  /*** Methods used in implementation of derived classes ***/
  /// Start processing of events
  void startProcessing();

  /// Terminate processing of events
  void terminateProcessing(const edm::EventSetup* = nullptr);

  /// Process event
  bool processEvent(const edm::Event&, const edm::EventSetup&);

  /// begin run
  void beginRunImpl(const edm::Run&, const edm::EventSetup&);

  /// end run
  void endRunImpl(const edm::Run&, const edm::EventSetup&);

  /// begin lumi block
  void beginLuminosityBlockImpl(const edm::LuminosityBlock&, const edm::EventSetup&);

  /// end lumi block
  void endLuminosityBlockImpl(const edm::LuminosityBlock&, const edm::EventSetup&);

  int nEvent() const { return nevent_; }

  /// Creates Geometry and Alignables of the Tracker and initializes the
  /// AlignmentAlgorithm @alignmentAlgo_
  void initAlignmentAlgorithm(const edm::EventSetup&, bool update = false);

  /// Steers activities after end of job, terminates the AlignmentAlgorithm
  /// @alignmentAlgo_
  bool finish();

  virtual bool getTrajTrackAssociationCollection(const edm::Event&, edm::Handle<TrajTrackAssociationCollection>&) = 0;
  virtual bool getBeamSpot(const edm::Event&, edm::Handle<reco::BeamSpot>&) = 0;
  virtual bool getTkFittedLasBeamCollection(const edm::Run&, edm::Handle<TkFittedLasBeamCollection>&) = 0;
  virtual bool getTsosVectorCollection(const edm::Run&, edm::Handle<TsosVectorCollection>&) = 0;
  virtual bool getAliClusterValueMap(const edm::Event&, edm::Handle<AliClusterValueMap>&) = 0;

  std::shared_ptr<TrackerGeometry> trackerGeometry_;
  std::shared_ptr<DTGeometry> muonDTGeometry_;
  std::shared_ptr<CSCGeometry> muonCSCGeometry_;
  const bool doTracker_, doMuon_, useExtras_;

  /// Map with tracks/trajectories
  const edm::InputTag tjTkAssociationMapTag_;

  /// BeamSpot
  const edm::InputTag beamSpotTag_;

  /// LAS beams in edm::Run (ignore if empty)
  const edm::InputTag tkLasBeamTag_;

  /// ValueMap containing associtaion cluster-flag
  const edm::InputTag clusterValueMapTag_;

private:
  /// Creates the choosen alignment algorithm
  void createAlignmentAlgorithm();

  /// Creates the monitors
  void createMonitors();

  /// Creates the calibrations
  void createCalibrations();

  /// Checks if one of the EventSetup-Records has changed
  bool setupChanged(const edm::EventSetup&);

  /// Initializes Beamspot @beamSpot_ of Alignables @alignableExtras_
  void initBeamSpot(const edm::Event&);

  /// Creates ideal geometry @trackerGeometry_ from IdealGeometryRecord
  void createGeometries(const edm::EventSetup&, const TrackerTopology*);

  /// Applies Alignments from Database (GlobalPositionRcd) to Geometry
  /// @trackerGeometry_
  void applyAlignmentsToDB(const edm::EventSetup&);

  /// Creates Alignables @alignableTracker_ from the previously loaded
  /// Geometry @trackerGeometry_
  void createAlignables(const TrackerTopology*, bool update = false);

  /// Creates the @alignmentParameterStore_, which manages all Alignables
  void buildParameterStore();

  /// Applies misalignment scenario to @alignableTracker_
  void applyMisalignment();

  /// Applies misalignment scenario to @alignableTracker_
  void simpleMisalignment(const align::Alignables&, const std::string&, float, float, bool);

  /// Applies Alignments, AlignmentErrors and SurfaceDeformations to
  /// @trackerGeometry_
  void applyAlignmentsToGeometry();

  /// Applies DB constants belonging to (Err)Rcd to Geometry, taking into
  /// account 'globalPosition' correction.
  template <class G, class Rcd, class ErrRcd>
  void applyDB(G*, const edm::EventSetup&, const AlignTransform&) const;

  /// Applies DB constants for SurfaceDeformations
  template <class G, class DeformationRcd>
  void applyDB(G*, const edm::EventSetup&) const;

  /// Reads in survey records
  void readInSurveyRcds(const edm::EventSetup&);

  /// Adds survey info to an Alignable
  void addSurveyInfo(Alignable*);

  /// Writes Alignments (i.e. Records) to database-file
  void storeAlignmentsToDB();

  /// Writes Alignments and AlignmentErrors for all sub detectors and the
  /// given run number
  void writeForRunRange(cond::Time_t);

  /// Writes Alignments and/or AlignmentErrors to DB for record names
  /// (removes *globalCoordinates before writing if non-null...).
  /// Takes over ownership of Alignments and AlignmentErrors.
  void writeDB(Alignments*,
               const std::string&,
               AlignmentErrorsExtended*,
               const std::string&,
               const AlignTransform*,
               cond::Time_t) const;

  /// Writes SurfaceDeformations (bows & kinks) to DB for given record name
  /// Takes over ownership of AlignmentSurfaceDeformations.
  void writeDB(AlignmentSurfaceDeformations*, const std::string&, cond::Time_t) const;

  template <typename T>
  bool hasParameter(const edm::ParameterSet&, const std::string& name);

  //========================== PRIVATE DATA ====================================
  //============================================================================

  /*** Alignment data ***/

  std::unique_ptr<AlignmentAlgorithmBase> alignmentAlgo_;
  Calibrations calibrations_;
  AlignmentMonitors monitors_;

  AlignmentParameterStore* alignmentParameterStore_{nullptr};
  AlignableTracker* alignableTracker_{nullptr};
  AlignableMuon* alignableMuon_{nullptr};
  AlignableExtras* alignableExtras_{nullptr};

  edm::Handle<reco::BeamSpot> beamSpot_;
  /// GlobalPositions that might be read from DB, nullptr otherwise
  std::unique_ptr<const Alignments> globalPositions_;

  const align::RunRanges uniqueRunRanges_;
  int nevent_{0};
  bool runAtPCL_{false};

  /*** Parameters from config-file ***/

  edm::ParameterSet config_;

  const int stNFixAlignables_;
  const double stRandomShift_, stRandomRotation_;
  const bool applyDbAlignment_, checkDbAlignmentValidity_;
  const bool doMisalignmentScenario_;
  const bool saveToDB_, saveApeToDB_, saveDeformationsToDB_;
  const bool useSurvey_;
  const bool enableAlignableUpdates_;

  /*** ESWatcher ***/

  edm::ESWatcher<IdealGeometryRecord> watchIdealGeometryRcd_;
  edm::ESWatcher<GlobalPositionRcd> watchGlobalPositionRcd_;

  edm::ESWatcher<TrackerAlignmentRcd> watchTrackerAlRcd_;
  edm::ESWatcher<TrackerAlignmentErrorExtendedRcd> watchTrackerAlErrorExtRcd_;
  edm::ESWatcher<TrackerSurfaceDeformationRcd> watchTrackerSurDeRcd_;

  edm::ESWatcher<DTAlignmentRcd> watchDTAlRcd_;
  edm::ESWatcher<DTAlignmentErrorExtendedRcd> watchDTAlErrExtRcd_;
  edm::ESWatcher<CSCAlignmentRcd> watchCSCAlRcd_;
  edm::ESWatcher<CSCAlignmentErrorExtendedRcd> watchCSCAlErrExtRcd_;

  edm::ESWatcher<TrackerSurveyRcd> watchTkSurveyRcd_;
  edm::ESWatcher<TrackerSurveyErrorExtendedRcd> watchTkSurveyErrExtRcd_;
  edm::ESWatcher<DTSurveyRcd> watchDTSurveyRcd_;
  edm::ESWatcher<DTSurveyErrorExtendedRcd> watchDTSurveyErrExtRcd_;
  edm::ESWatcher<CSCSurveyRcd> watchCSCSurveyRcd_;
  edm::ESWatcher<CSCSurveyErrorExtendedRcd> watchCSCSurveyErrExtRcd_;

  /*** Survey stuff ***/

  size_t surveyIndex_{0};
  const Alignments* surveyValues_{nullptr};
  const SurveyErrors* surveyErrors_{nullptr};

  /*** Status flags ***/
  bool isAlgoInitialized_{false};
  bool isDuringLoop_{false};  // -> needed to ensure correct behaviour in
                              //    both, EDLooper and standard framework
                              //    modules
  cond::Time_t firstRun_{cond::timeTypeSpecs[cond::runnumber].endValue};
};

template <class G, class Rcd, class ErrRcd>
void AlignmentProducerBase::applyDB(G* geometry,
                                    const edm::EventSetup& iSetup,
                                    const AlignTransform& globalCoordinates) const {
  // 'G' is the geometry class for that DB should be applied,
  // 'Rcd' is the record class for its Alignments
  // 'ErrRcd' is the record class for its AlignmentErrorsExtended
  // 'globalCoordinates' are global transformation for this geometry

  const Rcd& record = iSetup.get<Rcd>();
  if (checkDbAlignmentValidity_) {
    const edm::ValidityInterval& validity = record.validityInterval();
    const edm::IOVSyncValue first = validity.first();
    const edm::IOVSyncValue last = validity.last();
    if (first != edm::IOVSyncValue::beginOfTime() || last != edm::IOVSyncValue::endOfTime()) {
      throw cms::Exception("DatabaseError")
          << "@SUB=AlignmentProducerBase::applyDB"
          << "\nTrying to apply " << record.key().name() << " with multiple IOVs in tag.\n"
          << "Validity range is " << first.eventID().run() << " - " << last.eventID().run();
    }
  }

  edm::ESHandle<Alignments> alignments;
  record.get(alignments);

  edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
  iSetup.get<ErrRcd>().get(alignmentErrors);

  GeometryAligner aligner;
  aligner.applyAlignments<G>(geometry, &(*alignments), &(*alignmentErrors), globalCoordinates);
}

template <class G, class DeformationRcd>
void AlignmentProducerBase::applyDB(G* geometry, const edm::EventSetup& iSetup) const {
  // 'G' is the geometry class for that DB should be applied,
  // 'DeformationRcd' is the record class for its surface deformations

  const DeformationRcd& record = iSetup.get<DeformationRcd>();
  if (checkDbAlignmentValidity_) {
    const edm::ValidityInterval& validity = record.validityInterval();
    const edm::IOVSyncValue first = validity.first();
    const edm::IOVSyncValue last = validity.last();
    if (first != edm::IOVSyncValue::beginOfTime() || last != edm::IOVSyncValue::endOfTime()) {
      throw cms::Exception("DatabaseError")
          << "@SUB=AlignmentProducerBase::applyDB"
          << "\nTrying to apply " << record.key().name() << " with multiple IOVs in tag.\n"
          << "Validity range is " << first.eventID().run() << " - " << last.eventID().run();
    }
  }
  edm::ESHandle<AlignmentSurfaceDeformations> surfaceDeformations;
  record.get(surfaceDeformations);

  GeometryAligner aligner;
  aligner.attachSurfaceDeformations<G>(geometry, &(*surfaceDeformations));
}

#endif /* Alignment_CommonAlignmentProducer_AlignmentProducerBase_h */
