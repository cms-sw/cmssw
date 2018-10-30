#include "Alignment/CommonAlignmentProducer/interface/AlignmentProducerBase.h"

#include "Alignment/CommonAlignment/interface/AlignableExtras.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterBuilder.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/IntegratedCalibrationPluginFactory.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "Alignment/CommonAlignmentParametrization/interface/BeamSpotAlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/MuonAlignment/interface/MuonScenarioBuilder.h"
#include "Alignment/TrackerAlignment/interface/TrackerScenarioBuilder.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/SurveyError.h"
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"


//------------------------------------------------------------------------------
AlignmentProducerBase::AlignmentProducerBase(const edm::ParameterSet& config) :
  doTracker_{config.getUntrackedParameter<bool>("doTracker")},
  doMuon_{config.getUntrackedParameter<bool>("doMuon")},
  useExtras_{config.getUntrackedParameter<bool>("useExtras")},
  tjTkAssociationMapTag_{config.getParameter<edm::InputTag>("tjTkAssociationMapTag")},
  beamSpotTag_{config.getParameter<edm::InputTag>("beamSpotTag")},
  tkLasBeamTag_{config.getParameter<edm::InputTag>("tkLasBeamTag")},
  clusterValueMapTag_{config.getParameter<edm::InputTag>("hitPrescaleMapTag")},
  uniqueRunRanges_
  {align::makeUniqueRunRanges(config.getParameter<edm::VParameterSet>("RunRangeSelection"),
                              cond::timeTypeSpecs[cond::runnumber].beginValue)},
  config_{config},
  stNFixAlignables_{config.getParameter<int>("nFixAlignables")},
  stRandomShift_{config.getParameter<double>("randomShift")},
  stRandomRotation_{config.getParameter<double>("randomRotation")},
  applyDbAlignment_{config.getUntrackedParameter<bool>("applyDbAlignment")},
  checkDbAlignmentValidity_{config.getUntrackedParameter<bool>("checkDbAlignmentValidity")},
  doMisalignmentScenario_{config.getParameter<bool>("doMisalignmentScenario")},
  saveToDB_{config.getParameter<bool>("saveToDB")},
  saveApeToDB_{config.getParameter<bool>("saveApeToDB")},
  saveDeformationsToDB_{config.getParameter<bool>("saveDeformationsToDB")},
  useSurvey_{config.getParameter<bool>("useSurvey")},
  enableAlignableUpdates_{config.getParameter<bool>("enableAlignableUpdates")}
{
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducerBase::AlignmentProducerBase";

  const auto& algoConfig = config_.getParameterSet("algoConfig");
  if (hasParameter<bool>(config_, "runAtPCL")) {
    // configured in main config?
    runAtPCL_ = config_.getParameter<bool>("runAtPCL");

    if (hasParameter<bool>(algoConfig, "runAtPCL") &&
        (runAtPCL_ != algoConfig.getParameter<bool>("runAtPCL"))) {
      throw cms::Exception("BadConfig")
        << "Inconsistent settings for 'runAtPCL' in configuration of the "
        << "alignment producer and the alignment algorithm.";
    }

  } else if (hasParameter<bool>(algoConfig, "runAtPCL")) {
    // configured in algo config?
    runAtPCL_ = algoConfig.getParameter<bool>("runAtPCL");

  } else {
    // assume 'false' if it was not configured
    runAtPCL_ = false;
  }

  createAlignmentAlgorithm();
  createMonitors();
  createCalibrations();
}



//------------------------------------------------------------------------------
AlignmentProducerBase::~AlignmentProducerBase() noexcept(false)
{
  for (auto& iCal: calibrations_) delete iCal;

  delete alignmentParameterStore_;
  delete alignableExtras_;
  delete alignableTracker_;
  delete alignableMuon_;
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::startProcessing()
{
  if (isDuringLoop_) return;

  edm::LogInfo("Alignment")
    << "@SUB=AlignmentProducerBase::startProcessing"
    << "Begin";

  if (!isAlgoInitialized_) {
    throw cms::Exception("LogicError")
      << "@SUB=AlignmentProducerBase::startProcessing\n"
      << "Trying to start event processing before initializing the alignment "
      << "algorithm.";
  }

  nevent_ = 0;

  alignmentAlgo_->startNewLoop();

  // FIXME: Should this be done in algorithm::startNewLoop()??
  for (const auto& iCal: calibrations_) iCal->startNewLoop();
  for (const auto& monitor: monitors_) monitor->startingNewLoop();

  applyAlignmentsToGeometry();
  isDuringLoop_ = true;
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::terminateProcessing(const edm::EventSetup* setup)
{
  if (!isDuringLoop_) return;

  edm::LogInfo("Alignment")
    << "@SUB=AlignmentProducerBase::terminateProcessing"
    << "Terminating algorithm.";
  if (setup) {
    alignmentAlgo_->terminate(*setup);
  } else {
    alignmentAlgo_->terminate();
  }

  // FIXME: Should this be done in algorithm::terminate()??
  for (const auto& iCal:calibrations_) iCal->endOfLoop();
  for (const auto& monitor: monitors_) monitor->endOfLoop();

  isDuringLoop_ = false;
}


//------------------------------------------------------------------------------
bool
AlignmentProducerBase::processEvent(const edm::Event& event,
                                    const edm::EventSetup& setup)
{
  if (setupChanged(setup)) {
    edm::LogInfo("Alignment")
      << "@SUB=AlignmentProducerBase::processEvent"
      << "EventSetup-Record changed.";

    // updatable alignables are currently not used at PCL, but event setup
    // changes require a complete re-initialization
    if (runAtPCL_) {
      initAlignmentAlgorithm(setup, /* update = */ false);
    } else if (enableAlignableUpdates_) {
      initAlignmentAlgorithm(setup, /* update = */ true);
    }
  }

  initBeamSpot(event); // must happen every event and before incrementing 'nevent_'

  ++nevent_;  // must happen before the check below;
              // otherwise subsequent checks fail for "EmptySource"

  if (!alignmentAlgo_->processesEvents()) {
    edm::LogInfo("Alignment")
      << "@SUB=AlignmentProducerBase::processEvent"
      << "Skipping event. The current configuration of the alignment algorithm "
      << "does not need to process any events.";
    return false;
  }

  // reading in survey records
  readInSurveyRcds(setup);

  // Printout event number
  for ( int i=10; i<10000000; i*=10 ) {
    if ( nevent_<10*i && (nevent_%i)==0 ) {
      edm::LogInfo("Alignment")
        << "@SUB=AlignmentProducerBase::processEvent"
        << "Events processed: " << nevent_;
    }
  }

  // Retrieve trajectories and tracks from the event
  // -> merely skip if collection is empty
  edm::Handle<TrajTrackAssociationCollection> handleTrajTracksCollection;

  if (getTrajTrackAssociationCollection(event, handleTrajTracksCollection)) {

    // Form pairs of trajectories and tracks
    ConstTrajTrackPairs trajTracks;
    for (auto iter  = handleTrajTracksCollection->begin();
              iter != handleTrajTracksCollection->end();
            ++iter) {
      trajTracks.push_back(ConstTrajTrackPair(&(*(*iter).key), &(*(*iter).val)));
    }

    // Run the alignment algorithm with its input
    const AliClusterValueMap* clusterValueMapPtr{nullptr};
    if (!clusterValueMapTag_.encode().empty()) {
      edm::Handle<AliClusterValueMap> clusterValueMap;
      getAliClusterValueMap(event, clusterValueMap);
      clusterValueMapPtr = &(*clusterValueMap);
    }


    const AlignmentAlgorithmBase::EventInfo eventInfo{event.id(),
                                                      trajTracks,
                                                      *beamSpot_,
                                                      clusterValueMapPtr};
    alignmentAlgo_->run(setup, eventInfo);


    for (const auto& monitor: monitors_) {
      monitor->duringLoop(event, setup, trajTracks); // forward eventInfo?
    }
  } else {
    edm::LogError("Alignment")
      << "@SUB=AlignmentProducerBase::processEvent"
      << "No track collection found: skipping event";
  }

  return true;
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::beginRunImpl(const edm::Run& run, const edm::EventSetup& setup)
{
  const bool changed{setupChanged(setup)};
  if (changed) {
    edm::LogInfo("Alignment")
      << "@SUB=AlignmentProducerBase::beginRunImpl"
      << "EventSetup-Record changed.";

    // updatable alignables are currently not used at PCL, but event setup
    // changes require a complete re-initialization
    if (runAtPCL_) {
      initAlignmentAlgorithm(setup, /* update = */ false);
    } else if (enableAlignableUpdates_) {
      initAlignmentAlgorithm(setup, /* update = */ true);
    }
  }

  alignmentAlgo_->beginRun(run, setup,
                           changed && (runAtPCL_ || enableAlignableUpdates_));

  for (const auto& iCal:calibrations_) iCal->beginRun(run, setup);

  //store the first run analyzed to be used for setting the IOV (for PCL)
  if (firstRun_ > static_cast<cond::Time_t>(run.id().run())) {
    firstRun_ = static_cast<cond::Time_t>(run.id().run());
  }
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::endRunImpl(const edm::Run& run, const edm::EventSetup& setup)
{
  if (!tkLasBeamTag_.encode().empty()) {
    edm::Handle<TkFittedLasBeamCollection> lasBeams;
    edm::Handle<TsosVectorCollection> tsoses;
    getTkFittedLasBeamCollection(run, lasBeams);
    getTsosVectorCollection(run, tsoses);

    alignmentAlgo_->endRun(EndRunInfo(run.id(), &(*lasBeams), &(*tsoses)), setup);
  } else {
    edm::LogInfo("Alignment")
      << "@SUB=AlignmentProducerBase::endRunImpl"
      << "No Tk LAS beams to forward to algorithm.";
    alignmentAlgo_->endRun(EndRunInfo(run.id(), nullptr, nullptr), setup);
  }

}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::beginLuminosityBlockImpl(const edm::LuminosityBlock&,
                                                const edm::EventSetup& setup)
{
  // Do not forward edm::LuminosityBlock
  alignmentAlgo_->beginLuminosityBlock(setup);
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::endLuminosityBlockImpl(const edm::LuminosityBlock&,
                                              const edm::EventSetup& setup)
{
  // Do not forward edm::LuminosityBlock
  alignmentAlgo_->endLuminosityBlock(setup);
}



//------------------------------------------------------------------------------
void
AlignmentProducerBase::createAlignmentAlgorithm()
{
  auto algoConfig = config_.getParameter<edm::ParameterSet>("algoConfig");
  algoConfig.addUntrackedParameter("RunRangeSelection",
                                   config_.getParameter<edm::VParameterSet>("RunRangeSelection"));
  algoConfig.addUntrackedParameter<align::RunNumber>("firstIOV",
                                                     runAtPCL_ ?
                                                     1 :
                                                     uniqueRunRanges_.front().first);
  algoConfig.addUntrackedParameter("enableAlignableUpdates",
                                   enableAlignableUpdates_);

  const auto& algoName = algoConfig.getParameter<std::string>("algoName");
  alignmentAlgo_ = std::unique_ptr<AlignmentAlgorithmBase>
    {AlignmentAlgorithmPluginFactory::get()->create(algoName, algoConfig)};

  if (!alignmentAlgo_) {
    throw cms::Exception("BadConfig")
      << "Couldn't find the called alignment algorithm: " << algoName;
  }
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::createMonitors()
{
  const auto& monitorConfig = config_.getParameter<edm::ParameterSet>("monitorConfig");
  auto monitors = monitorConfig.getUntrackedParameter<std::vector<std::string> >("monitors");
  for (const auto& miter: monitors) {
    std::unique_ptr<AlignmentMonitorBase> newMonitor
      {AlignmentMonitorPluginFactory::get()
       ->create(miter, monitorConfig.getUntrackedParameterSet(miter))};

    if (!newMonitor) {
      throw cms::Exception("BadConfig") << "Couldn't find monitor named " << miter;
    }
    monitors_.emplace_back(std::move(newMonitor));
  }
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::createCalibrations()
{
  const auto& calibrations = config_.getParameter<edm::VParameterSet>("calibrations");
  for (const auto& iCalib: calibrations) {
    calibrations_.push_back(IntegratedCalibrationPluginFactory::get()
                            ->create(iCalib.getParameter<std::string>("calibrationName"),
                                     iCalib));
  }
}


//------------------------------------------------------------------------------
bool
AlignmentProducerBase::setupChanged(const edm::EventSetup& setup)
{
  bool changed{false};

  if (watchIdealGeometryRcd_.check(setup)) {
    changed = true;
  }

  if (watchGlobalPositionRcd_.check(setup)) {
    changed = true;
  }

  if (doTracker_) {
    if (watchTrackerAlRcd_.check(setup)) {
      changed = true;
    }

    if (watchTrackerAlErrorExtRcd_.check(setup)) {
      changed = true;
    }

    if (watchTrackerSurDeRcd_.check(setup)) {
      changed = true;
    }
  }

  if (doMuon_) {
    if (watchDTAlRcd_.check(setup)) {
      changed = true;
    }

    if (watchDTAlErrExtRcd_.check(setup)) {
      changed = true;
    }

    if (watchCSCAlRcd_.check(setup)) {
      changed = true;
    }

    if (watchCSCAlErrExtRcd_.check(setup)) {
      changed = true;
    }
  }

  /* TODO: ExtraAlignables: Which record(s) to check?
   *
   if (useExtras_) {}
  */

  return changed;
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::initAlignmentAlgorithm(const edm::EventSetup& setup,
                                              bool update)
{
  edm::LogInfo("Alignment")
    << "@SUB=AlignmentProducerBase::initAlignmentAlgorithm"
    << "Begin";

  auto isTrueUpdate = update && isAlgoInitialized_;

  // Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  setup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // Create the geometries from the ideal geometries
  createGeometries(setup, tTopo);

  applyAlignmentsToDB(setup);
  createAlignables(tTopo, isTrueUpdate);
  buildParameterStore();
  applyMisalignment();

  // Initialize alignment algorithm and integrated calibration and pass the
  // latter to algorithm
  edm::LogInfo("Alignment")
    << "@SUB=AlignmentProducerBase::initAlignmentAlgorithm"
    << "Initializing alignment algorithm.";
  alignmentAlgo_->initialize(setup,
                             alignableTracker_,
                             alignableMuon_,
                             alignableExtras_,
                             alignmentParameterStore_);

  // Not all algorithms support calibrations - so do not pass empty vector
  // and throw if non-empty and not supported:
  if (!calibrations_.empty()) {
    if (alignmentAlgo_->supportsCalibrations()) {
      alignmentAlgo_->addCalibrations(calibrations_);
    } else {
      throw cms::Exception("BadConfig")
        << "@SUB=AlignmentProducerBase::createCalibrations\n"
        << "Configured " << calibrations_.size() << " calibration(s) "
        << "for algorithm not supporting it.";
    }
  }

  isAlgoInitialized_ = true;

  applyAlignmentsToGeometry();

  if (!isTrueUpdate) {                // only needed the first time
    for (const auto& iCal: calibrations_) {
      iCal->beginOfJob(alignableTracker_, alignableMuon_, alignableExtras_);
    }
    for (const auto& monitor: monitors_) {
     monitor->beginOfJob(alignableTracker_, alignableMuon_, alignmentParameterStore_);
    }
  }
  startProcessing();            // needed if derived class is non-EDLooper-based
                                // has no effect, if called during loop

  edm::LogInfo("Alignment")
    << "@SUB=AlignmentProducerBase::initAlignmentAlgorithm"
    << "End";
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::initBeamSpot(const edm::Event& event)
{
  getBeamSpot(event, beamSpot_);

  if (nevent_== 0 && alignableExtras_) {
    edm::LogInfo("Alignment")
      << "@SUB=AlignmentProducerBase::initBeamSpot"
      << "Initializing AlignableBeamSpot";

    alignableExtras_->initializeBeamSpot(beamSpot_->x0(),
                                         beamSpot_->y0(),
                                         beamSpot_->z0(),
                                         beamSpot_->dxdz(),
                                         beamSpot_->dydz());
  }
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::createGeometries(const edm::EventSetup& iSetup,
                                        const TrackerTopology* tTopo)
{
  if (doTracker_) {
    edm::ESHandle<GeometricDet> geometricDet;
    iSetup.get<IdealGeometryRecord>().get(geometricDet);

    edm::ESHandle<PTrackerParameters> ptp;
    iSetup.get<PTrackerParametersRcd>().get(ptp);

    TrackerGeomBuilderFromGeometricDet trackerBuilder;

    trackerGeometry_ = std::shared_ptr<TrackerGeometry>
      (trackerBuilder.build(&(*geometricDet), *ptp, tTopo));
  }

  if (doMuon_) {
    edm::ESTransientHandle<DDCompactView> cpv;
    iSetup.get<IdealGeometryRecord>().get(cpv);
    edm::ESHandle<MuonDDDConstants> mdc;
    iSetup.get<MuonNumberingRecord>().get(mdc);
    DTGeometryBuilderFromDDD DTGeometryBuilder;
    CSCGeometryBuilderFromDDD CSCGeometryBuilder;
    muonDTGeometry_ = std::make_shared<DTGeometry>();
    DTGeometryBuilder.build(*muonDTGeometry_, &(*cpv), *mdc);
    muonCSCGeometry_ = std::make_shared<CSCGeometry>();
    CSCGeometryBuilder.build(*muonCSCGeometry_, &(*cpv), *mdc );
  }
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::applyAlignmentsToDB(const edm::EventSetup& setup)
{
  // Retrieve and apply alignments, if requested (requires z setup)
  if (applyDbAlignment_) {
    // we need GlobalPositionRcd - and have to keep track for later removal
    // before writing again to DB...

    edm::ESHandle<Alignments> globalAlignments;
    setup.get<GlobalPositionRcd>().get(globalAlignments);
    globalPositions_ = std::make_unique<Alignments>(*globalAlignments);

    if (doTracker_) {
      applyDB<TrackerGeometry,
              TrackerAlignmentRcd,
              TrackerAlignmentErrorExtendedRcd>
        (trackerGeometry_.get(), setup,
         align::DetectorGlobalPosition(*globalPositions_, DetId(DetId::Tracker))
         );

      applyDB<TrackerGeometry,
              TrackerSurfaceDeformationRcd>(trackerGeometry_.get(), setup);
    }

    if (doMuon_) {
      applyDB<DTGeometry,
              DTAlignmentRcd,
              DTAlignmentErrorExtendedRcd>
        (muonDTGeometry_.get(), setup,
         align::DetectorGlobalPosition(*globalPositions_, DetId(DetId::Muon))
         );

      applyDB<CSCGeometry,
              CSCAlignmentRcd,
              CSCAlignmentErrorExtendedRcd>
        (muonCSCGeometry_.get(), setup,
         align::DetectorGlobalPosition(*globalPositions_, DetId(DetId::Muon))
         );
    }
  }
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::createAlignables(const TrackerTopology* tTopo, bool update)
{
  if (doTracker_) {
    if (update) {
      alignableTracker_->update(trackerGeometry_.get(), tTopo);
    } else {
      alignableTracker_ = new AlignableTracker(trackerGeometry_.get(), tTopo);
    }
  }

  if (doMuon_) {
    if (update) {
      alignableMuon_->update(muonDTGeometry_.get(), muonCSCGeometry_.get());
    } else {
      alignableMuon_ = new AlignableMuon(muonDTGeometry_.get(), muonCSCGeometry_.get());
    }
  }

  if (useExtras_) {
    if (update) {
      // FIXME: Requires further code changes to track beam spot condition changes
    } else {
      alignableExtras_ = new AlignableExtras();
    }
  }
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::buildParameterStore()
{
  // Create alignment parameter builder
  edm::LogInfo("Alignment")
    << "@SUB=AlignmentProducerBase::buildParameterStore"
    << "Creating AlignmentParameterBuilder";

  const auto& alParamBuildCfg =
    config_.getParameter<edm::ParameterSet>("ParameterBuilder");
  const auto& alParamStoreCfg =
    config_.getParameter<edm::ParameterSet>("ParameterStore");

  AlignmentParameterBuilder alignmentParameterBuilder{alignableTracker_,
                                                      alignableMuon_,
                                                      alignableExtras_,
                                                      alParamBuildCfg};

  // Fix alignables if requested
  if (stNFixAlignables_ > 0) {
    alignmentParameterBuilder.fixAlignables(stNFixAlignables_);
  }

  // Get list of alignables
  const auto& alignables = alignmentParameterBuilder.alignables();
  edm::LogInfo("Alignment")
    << "@SUB=AlignmentProducerBase::buildParameterStore"
    << "got " << alignables.size() << " alignables";

  // Create AlignmentParameterStore
  alignmentParameterStore_ =
    new AlignmentParameterStore(alignables, alParamStoreCfg);
  edm::LogInfo("Alignment")
    << "@SUB=AlignmentProducerBase::buildParameterStore"
    << "AlignmentParameterStore created!";
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::applyMisalignment()
{
  // Apply misalignment scenario to alignable tracker and muon if requested
  // WARNING: this assumes scenarioConfig can be passed to both muon and tracker

  if (doMisalignmentScenario_ && (doTracker_ || doMuon_)) {
    edm::LogInfo("Alignment")
      << "@SUB=AlignmentProducerBase::applyMisalignment"
      << "Applying misalignment scenario to "
      << (doTracker_ ? "tracker" : "")
      << (doMuon_    ? (doTracker_ ? " and muon" : "muon") : ".");

    const auto& scenarioConfig = config_.getParameterSet("MisalignmentScenario");

    if (doTracker_) {
      TrackerScenarioBuilder scenarioBuilder(alignableTracker_);
      scenarioBuilder.applyScenario(scenarioConfig);
    }
    if (doMuon_) {
      MuonScenarioBuilder muonScenarioBuilder(alignableMuon_);
      muonScenarioBuilder.applyScenario(scenarioConfig);
    }

  } else {
    edm::LogInfo("Alignment")
      << "@SUB=AlignmentProducerBase::applyMisalignment"
      << "NOT applying misalignment scenario!";
  }

  // Apply simple misalignment
  const auto& sParSel =
    config_.getParameter<std::string>("parameterSelectorSimple");
  simpleMisalignment(alignmentParameterStore_->alignables(), sParSel,
                     stRandomShift_, stRandomRotation_, true);
}


// ----------------------------------------------------------------------------
void
AlignmentProducerBase::simpleMisalignment(const align::Alignables &alivec,
                                          const std::string &selection,
                                          float shift, float rot, bool local)
{

  std::ostringstream output; // collecting output

  if (shift > 0. || rot > 0.) {
    output << "Adding random flat shift of max size " << shift
           << " and adding random flat rotation of max size " << rot <<" to ";

    std::vector<bool> commSel(0);
    if (selection != "-1") {
      AlignmentParameterSelector aSelector(nullptr,nullptr); // no alignable needed here...
      const std::vector<char> cSel(aSelector.convertParamSel(selection));
      if (cSel.size() < RigidBodyAlignmentParameters::N_PARAM) {
        throw cms::Exception("BadConfig")
          << "[AlignmentProducerBase::simpleMisalignment_]\n"
          << "Expect selection string '" << selection << "' to be at least of length "
          << RigidBodyAlignmentParameters::N_PARAM << " or to be '-1'.\n"
          << "(Most probably you have to adjust the parameter 'parameterSelectorSimple'.)";
      }
      for (const auto& cIter: cSel) {
        commSel.push_back(cIter == '0' ? false : true);
      }
      output << "parameters defined by (" << selection
             << "), representing (x,y,z,alpha,beta,gamma),";
    } else {
      output << "the active parameters of each alignable,";
    }
    output << " in " << (local ? "local" : "global") << " frame.";

    for (const auto& ali: alivec) {
      std::vector<bool> mysel(commSel.empty() ? ali->alignmentParameters()->selector() : commSel);

      if (std::abs(shift)>0.00001) {
        double s0 = 0., s1 = 0., s2 = 0.;
        if (mysel[RigidBodyAlignmentParameters::dx]) s0 = shift * double(random()%1000-500)/500.;
        if (mysel[RigidBodyAlignmentParameters::dy]) s1 = shift * double(random()%1000-500)/500.;
        if (mysel[RigidBodyAlignmentParameters::dz]) s2 = shift * double(random()%1000-500)/500.;

        if (local) ali->move(ali->surface().toGlobal(align::LocalVector(s0,s1,s2)));
        else       ali->move(align::GlobalVector(s0,s1,s2));

        //AlignmentPositionError ape(dx,dy,dz);
        //ali->addAlignmentPositionError(ape);
      }

      if (std::abs(rot)>0.00001) {
        align::EulerAngles r(3);
        if (mysel[RigidBodyAlignmentParameters::dalpha]) r(1)=rot*double(random()%1000-500)/500.;
        if (mysel[RigidBodyAlignmentParameters::dbeta])  r(2)=rot*double(random()%1000-500)/500.;
        if (mysel[RigidBodyAlignmentParameters::dgamma]) r(3)=rot*double(random()%1000-500)/500.;

        const align::RotationType mrot = align::toMatrix(r);
        if (local) ali->rotateInLocalFrame(mrot);
        else       ali->rotateInGlobalFrame(mrot);

        //ali->addAlignmentPositionErrorFromRotation(mrot);
      }
    } // end loop on alignables
  } else {
    output << "No simple misalignment added!";
  }
  edm::LogInfo("Alignment")
    << "@SUB=AlignmentProducerBase::simpleMisalignment" << output.str();
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::applyAlignmentsToGeometry()
{
  edm::LogInfo("Alignment")
    << "@SUB=AlignmentProducerBase::applyAlignmentsToGeometry"
    << "Now physically apply alignments to  geometry...";

  // Propagate changes to reconstruction geometry (from initialisation or iteration)
  GeometryAligner aligner;

  if (doTracker_) {
    if (!alignableTracker_) {
      throw cms::Exception("LogicError")
        << "@SUB=AlignmentProducerBase::applyAlignmentsToGeometry\n"
        << "Trying to apply tracker alignment before creating it.";
    }

    std::unique_ptr<Alignments> alignments{alignableTracker_->alignments()};
    std::unique_ptr<AlignmentErrorsExtended>
      alignmentErrExt{alignableTracker_->alignmentErrors()};
    std::unique_ptr<AlignmentSurfaceDeformations>
      aliDeforms{alignableTracker_->surfaceDeformations()};

    aligner.applyAlignments(trackerGeometry_.get(), alignments.get(),
                            alignmentErrExt.get(), AlignTransform());
    aligner.attachSurfaceDeformations(trackerGeometry_.get(), aliDeforms.get());
  }

  if (doMuon_) {
    if (!alignableMuon_) {
      throw cms::Exception("LogicError")
        << "@SUB=AlignmentProducerBase::applyAlignmentsToGeometry\n"
        << "Trying to apply muon alignment before creating it.";
    }

    std::unique_ptr<Alignments> dtAlignments{alignableMuon_->dtAlignments()};
    std::unique_ptr<Alignments> cscAlignments{alignableMuon_->cscAlignments()};

    std::unique_ptr<AlignmentErrorsExtended>
      dtAlignmentErrExt{alignableMuon_->dtAlignmentErrorsExtended()};
    std::unique_ptr<AlignmentErrorsExtended>
      cscAlignmentErrExt{alignableMuon_->cscAlignmentErrorsExtended()};

    aligner.applyAlignments(muonDTGeometry_.get(), dtAlignments.get(),
                            dtAlignmentErrExt.get(), AlignTransform());
    aligner.applyAlignments(muonCSCGeometry_.get(), cscAlignments.get(),
                            cscAlignmentErrExt.get(), AlignTransform());
  }
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::readInSurveyRcds(const edm::EventSetup& iSetup)
{

  // Get Survey Rcds and add Survey Info
  if ( doTracker_ && useSurvey_ ){
    bool tkSurveyBool = watchTkSurveyRcd_.check(iSetup);
    bool tkSurveyErrBool = watchTkSurveyErrExtRcd_.check(iSetup);
    edm::LogInfo("Alignment") << "watcher tksurveyrcd: " << tkSurveyBool;
    edm::LogInfo("Alignment") << "watcher tksurveyerrrcd: " << tkSurveyErrBool;
    if ( tkSurveyBool || tkSurveyErrBool){

      edm::LogInfo("Alignment") << "ADDING THE SURVEY INFORMATION";
      edm::ESHandle<Alignments> surveys;
      edm::ESHandle<SurveyErrors> surveyErrors;

      iSetup.get<TrackerSurveyRcd>().get(surveys);
      iSetup.get<TrackerSurveyErrorExtendedRcd>().get(surveyErrors);

      surveyIndex_  = 0;
      surveyValues_ = &*surveys;
      surveyErrors_ = &*surveyErrors;
      addSurveyInfo(alignableTracker_);
    }
  }

  if ( doMuon_ && useSurvey_) {
    bool DTSurveyBool = watchTkSurveyRcd_.check(iSetup);
    bool DTSurveyErrBool = watchTkSurveyErrExtRcd_.check(iSetup);
    bool CSCSurveyBool = watchTkSurveyRcd_.check(iSetup);
    bool CSCSurveyErrBool = watchTkSurveyErrExtRcd_.check(iSetup);

    if ( DTSurveyBool || DTSurveyErrBool || CSCSurveyBool || CSCSurveyErrBool ){
      edm::ESHandle<Alignments> dtSurveys;
      edm::ESHandle<SurveyErrors> dtSurveyErrors;
      edm::ESHandle<Alignments> cscSurveys;
      edm::ESHandle<SurveyErrors> cscSurveyErrors;

      iSetup.get<DTSurveyRcd>().get(dtSurveys);
      iSetup.get<DTSurveyErrorExtendedRcd>().get(dtSurveyErrors);
      iSetup.get<CSCSurveyRcd>().get(cscSurveys);
      iSetup.get<CSCSurveyErrorExtendedRcd>().get(cscSurveyErrors);

      surveyIndex_  = 0;
      surveyValues_ = &*dtSurveys;
      surveyErrors_ = &*dtSurveyErrors;
      const auto& barrels = alignableMuon_->DTBarrel();
      for (const auto& barrel: barrels) addSurveyInfo(barrel);

      surveyIndex_  = 0;
      surveyValues_ = &*cscSurveys;
      surveyErrors_ = &*cscSurveyErrors;
      const auto& endcaps = alignableMuon_->CSCEndcaps();
      for (const auto& endcap: endcaps) addSurveyInfo(endcap);

    }
  }

}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::addSurveyInfo(Alignable* ali)
{
  const auto& comps = ali->components();

  for (const auto& comp: comps) addSurveyInfo(comp);

  const SurveyError& error = surveyErrors_->m_surveyErrors[surveyIndex_];

  if ( ali->id() != error.rawId() ||
       ali->alignableObjectId() != error.structureType() )
    {
      throw cms::Exception("DatabaseError")
        << "Error reading survey info from DB. Mismatched id!";
    }

  const auto& pos = surveyValues_->m_align[surveyIndex_].translation();
  const auto& rot = surveyValues_->m_align[surveyIndex_].rotation();

  AlignableSurface surf(align::PositionType(pos.x(), pos.y(), pos.z()),
                        align::RotationType(rot.xx(), rot.xy(), rot.xz(),
                                            rot.yx(), rot.yy(), rot.yz(),
                                            rot.zx(), rot.zy(), rot.zz()));

  surf.setWidth(ali->surface().width());
  surf.setLength(ali->surface().length());

  ali->setSurvey(new SurveyDet(surf, error.matrix()));

  ++surveyIndex_;
}


//------------------------------------------------------------------------------
bool
AlignmentProducerBase::finish()
{

  for (const auto& monitor: monitors_) monitor->endOfJob();

  if (alignmentAlgo_->processesEvents() && nevent_ == 0) {
    return false;
  }

  if (saveToDB_ || saveApeToDB_ || saveDeformationsToDB_) {
    if (alignmentAlgo_->storeAlignments()) storeAlignmentsToDB();
  } else {
    edm::LogInfo("Alignment")
      << "@SUB=AlignmentProducerBase::finish"
      << "No payload to be stored!";
  }

  // takes care of storing output of calibrations, but needs to be called only
  // after 'storeAlignmentsToDB()'
  for (const auto& iCal:calibrations_) iCal->endOfJob();

  return true;
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::storeAlignmentsToDB()
{
  const auto runRangeSelectionVPSet =
    config_.getParameterSetVector("RunRangeSelection");

  // handle PCL use case
  const auto& uniqueRunRanges =
    (runAtPCL_ ?
     align::makeUniqueRunRanges(runRangeSelectionVPSet, firstRun_) :
     uniqueRunRanges_);

  std::vector<AlgebraicVector> beamSpotParameters;

  for (const auto& iRunRange: uniqueRunRanges) {

    alignmentAlgo_->setParametersForRunRange(iRunRange);

    // Save alignments to database
    if (saveToDB_ || saveApeToDB_ || saveDeformationsToDB_) {
      writeForRunRange(iRunRange.first);
    }

    // Deal with extra alignables, e.g. beam spot
    if (alignableExtras_) {
      auto& alis = alignableExtras_->beamSpot();
      if (!alis.empty()) {
        auto beamSpotAliPars =
          dynamic_cast<BeamSpotAlignmentParameters*>(alis[0]->alignmentParameters());
        if (!beamSpotAliPars) {
          throw cms::Exception("LogicError")
            << "@SUB=AlignmentProducerBase::storeAlignmentsToDB\n"
            << "First alignable of alignableExtras_ does not have "
            << "'BeamSpotAlignmentParameters', while it should have.";
        }

        beamSpotParameters.push_back(beamSpotAliPars->parameters());
      }
    }
  }

  if (alignableExtras_) {
    std::ostringstream bsOutput;

    auto itPar = beamSpotParameters.cbegin();
    for (auto iRunRange = uniqueRunRanges.cbegin();
         iRunRange != uniqueRunRanges.cend();
         ++iRunRange, ++itPar) {
      bsOutput << "Run range: " << (*iRunRange).first << " - " << (*iRunRange).second << "\n";
      bsOutput << "  Displacement: x=" << (*itPar)[0] << ", y=" << (*itPar)[1] << "\n";
      bsOutput << "  Slope: dx/dz=" << (*itPar)[2] << ", dy/dz=" << (*itPar)[3] << "\n";
    }

    edm::LogInfo("Alignment")
      << "@SUB=AlignmentProducerBase::storeAlignmentsToDB"
      << "Parameters for alignable beamspot:\n" << bsOutput.str();
  }
}


//------------------------------------------------------------------------------
void
AlignmentProducerBase::writeForRunRange(cond::Time_t time)
{
  if (doTracker_) { // first tracker
    const AlignTransform* trackerGlobal{nullptr}; // will be 'removed' from constants
    if (globalPositions_) { // i.e. applied before in applyDB
      trackerGlobal = &align::DetectorGlobalPosition(*globalPositions_,
                                                     DetId(DetId::Tracker));
    }

    auto alignments = alignableTracker_->alignments();
    auto alignmentErrors = alignableTracker_->alignmentErrors();
    this->writeDB(alignments, "TrackerAlignmentRcd",
                  alignmentErrors, "TrackerAlignmentErrorExtendedRcd", trackerGlobal,
                  time);

    // Save surface deformations to database
    if (saveDeformationsToDB_) {
      auto alignmentSurfaceDeformations = alignableTracker_->surfaceDeformations();
      this->writeDB(alignmentSurfaceDeformations, "TrackerSurfaceDeformationRcd", time);
    }
  }

  if (doMuon_) { // now muon
    const AlignTransform* muonGlobal{nullptr}; // will be 'removed' from constants
    if (globalPositions_) { // i.e. applied before in applyDB
      muonGlobal = &align::DetectorGlobalPosition(*globalPositions_,
                                                  DetId(DetId::Muon));
    }
    // Get alignments+errors, first DT - ownership taken over by writeDB(..), so no delete
    auto alignments = alignableMuon_->dtAlignments();
    auto alignmentErrors = alignableMuon_->dtAlignmentErrorsExtended();
    this->writeDB(alignments, "DTAlignmentRcd",
                  alignmentErrors, "DTAlignmentErrorExtendedRcd", muonGlobal,
                  time);

    // Get alignments+errors, now CSC - ownership taken over by writeDB(..), so no delete
    alignments = alignableMuon_->cscAlignments();
    alignmentErrors = alignableMuon_->cscAlignmentErrorsExtended();
    this->writeDB(alignments, "CSCAlignmentRcd",
                  alignmentErrors, "CSCAlignmentErrorExtendedRcd", muonGlobal,
                  time);
  }
}


//------------------------------------------------------------------------------
void AlignmentProducerBase::writeDB(Alignments* alignments,
                                    const std::string& alignRcd,
                                    AlignmentErrorsExtended* alignmentErrors,
                                    const std::string& errRcd,
                                    const AlignTransform* globalCoordinates,
                                    cond::Time_t time) const
{
  Alignments* tempAlignments = alignments;
  AlignmentErrorsExtended* tempAlignmentErrorsExtended = alignmentErrors;

  // Call service
  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (!poolDb.isAvailable()) { // Die if not available
    delete tempAlignments;      // promised to take over ownership...
    delete tempAlignmentErrorsExtended; // dito
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
  }

  if (globalCoordinates  // happens only if (applyDbAlignment_ == true)
      && globalCoordinates->transform() != AlignTransform::Transform::Identity) {

    tempAlignments = new Alignments();            // temporary storage for
    tempAlignmentErrorsExtended = new AlignmentErrorsExtended();  // final alignments and errors

    GeometryAligner aligner;
    aligner.removeGlobalTransform(alignments, alignmentErrors,
                                  *globalCoordinates,
                                  tempAlignments, tempAlignmentErrorsExtended);

    delete alignments;       // have to delete original alignments
    delete alignmentErrors;  // same thing for the errors

    edm::LogInfo("Alignment")
      << "@SUB=AlignmentProducerBase::writeDB"
      << "globalCoordinates removed from alignments (" << alignRcd
      << ") and errors (" << alignRcd << ").";
  }

  if (saveToDB_) {
    edm::LogInfo("Alignment") << "Writing Alignments for run " << time
                              << " to " << alignRcd << ".";
    poolDb->writeOne<Alignments>(tempAlignments, time, alignRcd);
  } else { // poolDb->writeOne(..) takes over 'alignments' ownership,...
    delete tempAlignments; // ...otherwise we have to delete, as promised!
  }

  if (saveApeToDB_) {
    edm::LogInfo("Alignment") << "Writing AlignmentErrorsExtended for run " << time
                              << " to " << errRcd << ".";
    poolDb->writeOne<AlignmentErrorsExtended>(tempAlignmentErrorsExtended, time, errRcd);
  } else { // poolDb->writeOne(..) takes over 'alignmentErrors' ownership,...
    delete tempAlignmentErrorsExtended; // ...otherwise we have to delete, as promised!
  }
}


//------------------------------------------------------------------------------
void AlignmentProducerBase::writeDB(AlignmentSurfaceDeformations* alignmentSurfaceDeformations,
                                    const std::string& surfaceDeformationRcd,
                                    cond::Time_t time) const
{
  // Call service
  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (!poolDb.isAvailable()) { // Die if not available
    delete alignmentSurfaceDeformations; // promised to take over ownership...
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
  }

  if (saveDeformationsToDB_) {
    edm::LogInfo("Alignment") << "Writing AlignmentSurfaceDeformations for run " << time
                              << " to " << surfaceDeformationRcd  << ".";
    poolDb->writeOne<AlignmentSurfaceDeformations>(alignmentSurfaceDeformations, time,
                                                   surfaceDeformationRcd);
  } else { // poolDb->writeOne(..) takes over 'surfaceDeformation' ownership,...
    delete alignmentSurfaceDeformations; // ...otherwise we have to delete, as promised!
  }
}


//------------------------------------------------------------------------------
template<typename T>
bool
AlignmentProducerBase::hasParameter(const edm::ParameterSet& config,
                                    const std::string& name)
{
  try {
    config.getParameter<T>(name);
  } catch (const edm::Exception& e) {
    if (e.categoryCode() == edm::errors::Configuration) {
      if (e.message().find("MissingParameter") != std::string::npos) {
        return false;
      } else {
        throw;
      }
    } else {
      throw;
    }
  }

  return true;
}
