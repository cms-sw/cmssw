#include "TrackerAlignmentProducerForPCL.h"

//#include "FWCore/Framework/interface/LooperFactory.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterBuilder.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

// System include files
#include <memory>
#include <sstream>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/Utilities/interface/Parse.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"

// Tracking, LAS and cluster flag map (fwd is enough!)
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Alignment/interface/AliClusterValueMapFwd.h"
#include "DataFormats/Alignment/interface/TkFittedLasBeamCollectionFwd.h"
#include "Alignment/LaserAlignment/interface/TsosVectorCollection.h"

// Alignment
#include "CondFormats/Alignment/interface/SurveyErrors.h"
#include "Alignment/TrackerAlignment/interface/TrackerScenarioBuilder.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/BeamSpotAlignmentParameters.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/IntegratedCalibrationPluginFactory.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"

TrackerAlignmentProducerForPCL::TrackerAlignmentProducerForPCL(const edm::ParameterSet &config) :
  theAlignmentAlgo(0), //theAlignmentAlgo(std::nullptr_t),
  theAlignmentParameterStore(0), //theAlignmentParameterStore(std::nullptr_t),
  theParameterSet(config),
  theAlignableExtras(0),
  theAlignableTracker(0),
  // theTracker
  globalPositions(0),
  nevent_(0),
  doTracker_(true),

  stNFixAlignables_        (config.getParameter<int>("nFixAlignables")),
  stRandomShift_           (config.getParameter<double>("randomShift")),
  stRandomRotation_        (config.getParameter<double>("randomRotation")),
  doMisalignmentScenario_  (config.getParameter<bool>("doMisalignmentScenario")),
  saveToDB                 (config.getParameter<bool>("saveToDB")),
  saveApeToDB              (config.getParameter<bool>("saveApeToDB")),
  saveDeformationsToDB     (config.getParameter<bool>("saveDeformationsToDB")),

  applyDbAlignment_        (config.getUntrackedParameter<bool>("applyDbAlignment")),
  checkDbAlignmentValidity_(config.getUntrackedParameter<bool>("checkDbAlignmentValidity")),
  useExtras_               (config.getUntrackedParameter<bool>("useExtras")),
  useSurvey_               (config.getParameter<bool>("useSurvey")),

  tjTkAssociationMapTag_   (config.getParameter<edm::InputTag>("tjTkAssociationMapTag")),
  beamSpotTag_             (config.getParameter<edm::InputTag>("beamSpotTag")),
  tkLasBeamTag_            (config.getParameter<edm::InputTag>("tkLasBeamTag")),
  clusterValueMapTag_      (config.getParameter<edm::InputTag>("hitPrescaleMapTag")) {

  // ESProducer method ?
  //setWhatProduced(this, &AlignmentProducer::produceTracker);
  //std::cout << "TrackerAlignmentProducerForPCL::constructor " <<std::endl;
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);

  // Creating the alignment algorithm
  edm::ParameterSet algoConfig    = config.getParameter<edm::ParameterSet>("algoConfig");
  edm::VParameterSet iovSelection = config.getParameter<edm::VParameterSet>("RunRangeSelection");
  algoConfig.addUntrackedParameter<edm::VParameterSet>("RunRangeSelection", iovSelection);

  std::string algoName = algoConfig.getParameter<std::string>("algoName");
  theAlignmentAlgo = AlignmentAlgorithmPluginFactory::get()->create(algoName, algoConfig);

  if (!theAlignmentAlgo) {
    throw cms::Exception("BadConfig") << "Couldn't find the called alignment algorithm" << algoName;
  }

  // Finally create integrated calibrations:
  edm::VParameterSet calibrations = config.getParameter<edm::VParameterSet>("calibrations");
  for (auto iCalib = calibrations.begin(); iCalib != calibrations.end(); ++iCalib) {
    theCalibrations.push_back(IntegratedCalibrationPluginFactory::get()->create(
      iCalib->getParameter<std::string>("calibrationName"), *iCalib)
    );
  }
}

TrackerAlignmentProducerForPCL::~TrackerAlignmentProducerForPCL() {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);

  delete theAlignmentAlgo;

  // delete integrated calibration pointed to by (*iCal)
  for (auto iCal = theCalibrations.begin(); iCal != theCalibrations.end(); ++iCal) {
    delete *iCal;
  }

  delete theAlignmentParameterStore;
  delete theAlignableExtras;
  delete theAlignableTracker;
  delete globalPositions;
}



/************************************
 *   PUBLIC METHOD IMPLEMENTATION   *
 ************************************/

void TrackerAlignmentProducerForPCL::beginJob() {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called UNUSED \n", __FUNCTION__, __FILE__);
  // -> init();
}

void TrackerAlignmentProducerForPCL::endJob() {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);

}

void TrackerAlignmentProducerForPCL::analyze(const edm::Event&      event,
                                             const edm::EventSetup& setup) {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);
  edm::LogInfo("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::analyze";
  //std::cout << "TrackerAlignmentProducerForPCL::analyze " <<std::endl;


  // if (nevent_ == 0) {
  //   init(setup);
  // }
  ++nevent_;

  //FIXME: what about this one? Shall this one be moved in the beginRun???
// reading in survey records
  readInSurveyRcds(setup);

  // Printout event number
  /*  ???
  for (int i=10; i<10000000; i*=10 ) {
    if (nevent_ < 10*i && (nevent_%i) == 0) {
      edm::LogInfo("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::analyze"
                                << "Events processed: " << nevent_;
    }
  }
  */

  // Retrieve trajectories and tracks from the event
  // -> merely skip if collection is empty
  edm::Handle<TrajTrackAssociationCollection> m_TrajTracksMap;

  // TODO: getByLabel -> getByToken
  //if (event.getByToken())
  if (event.getByLabel(tjTkAssociationMapTag_, m_TrajTracksMap)) {
    // Form pairs of trajectories and tracks
    ConstTrajTrackPairCollection trajTracks;
    for ( TrajTrackAssociationCollection::const_iterator iPair = m_TrajTracksMap->begin();
          iPair != m_TrajTracksMap->end(); ++iPair) {
      trajTracks.push_back( ConstTrajTrackPair( &(*(*iPair).key), &(*(*iPair).val) ) );
    }

    // TODO: This should be in the constructor / beginJob method
    edm::Handle<reco::BeamSpot> theBeamSpot;
    event.getByLabel(beamSpotTag_, theBeamSpot);
    if (theAlignableExtras) {
      edm::LogInfo("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::analyze"
                                << "initializing AlignableBeamSpot";
      theAlignableExtras->initializeBeamSpot(theBeamSpot->x0(), theBeamSpot->y0(), theBeamSpot->z0(),
                                             theBeamSpot->dxdz(), theBeamSpot->dydz());
    }

    // Run the alignment algorithm with its input
    const AliClusterValueMap* clusterValueMapPtr = 0;
    //check that the input tag is not empty
    if (clusterValueMapTag_.encode().size()) {
      edm::Handle<AliClusterValueMap> clusterValueMap;
      event.getByLabel(clusterValueMapTag_, clusterValueMap);
      clusterValueMapPtr = &(*clusterValueMap);
    }

    const AlignmentAlgorithmBase::EventInfo eventInfo(event.id(), trajTracks, *theBeamSpot,
                                                      clusterValueMapPtr);
    theAlignmentAlgo->run(setup, eventInfo);

  } else {
    edm::LogError("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::analyze"
                               << "No track collection found: skipping event";
  }
}

void TrackerAlignmentProducerForPCL::beginRun(const edm::Run& run, const edm::EventSetup& setup) {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);
  theAlignmentAlgo->beginRun(setup);
  init(setup);
}

void TrackerAlignmentProducerForPCL::endRun(const edm::Run& run, const edm::EventSetup& setup) {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);

  // call with or without las beam info...
  typedef AlignmentAlgorithmBase::EndRunInfo EndRunInfo;
  // if non-empty InputTag
  if (tkLasBeamTag_.encode().size()) {
    edm::Handle<TkFittedLasBeamCollection> lasBeams;
    edm::Handle<TsosVectorCollection> tsoses;
    run.getByLabel(tkLasBeamTag_, lasBeams);
    run.getByLabel(tkLasBeamTag_, tsoses);

    theAlignmentAlgo->endRun(EndRunInfo(run.id(), &(*lasBeams), &(*tsoses)), setup);

  } else {
    edm::LogInfo("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::endRun"
                  << "No Tk LAS beams to forward to algorithm.";
    theAlignmentAlgo->endRun(EndRunInfo(run.id(), 0, 0), setup);
  }

  finish();

}

void TrackerAlignmentProducerForPCL::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock,
                                                   const edm::EventSetup& setup) {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);
  theAlignmentAlgo->beginLuminosityBlock(setup); // do not forward edm::LuminosityBlock
}

void TrackerAlignmentProducerForPCL::endLuminosityBlock(const edm::LuminosityBlock& lumiBlock,
                                                 const edm::EventSetup& setup) {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);
  theAlignmentAlgo->endLuminosityBlock(setup); // do not forward edm::LuminosityBlock
}



/*************************************
 *   PRIVATE METHOD IMPLEMENTATION   *
 *************************************/

void TrackerAlignmentProducerForPCL::init(const edm::EventSetup& setup) {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);
  edm::LogInfo("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::init";



  /* 1) Former: AlignmentProducer::beginOfJob(const edm::EventSetup& setup) */

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  setup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // Create the geometries from the ideal geometries (first time only)
  createGeometries(setup);

  // Retrieve and apply alignments, if requested (requires DB setup)
  if (applyDbAlignment_) {
    // we need GlobalPositionRcd - and have to keep track for later removal
    // before writing again to DB...
    edm::ESHandle<Alignments> globalPositionRcd;
    setup.get<GlobalPositionRcd>().get(globalPositionRcd);
    globalPositions = new Alignments(*globalPositionRcd);

    applyDB<TrackerGeometry, TrackerAlignmentRcd, TrackerAlignmentErrorExtendedRcd>(
        &(*theTracker),
        setup,
        align::DetectorGlobalPosition(*globalPositions, DetId(DetId::Tracker))
    );

    applyDB<TrackerGeometry, TrackerSurfaceDeformationRcd>(
        &(*theTracker),
        setup
    );
  }

  // Create alignable tracker
  theAlignableTracker = new AlignableTracker(&(*theTracker), tTopo);

  if (useExtras_) {
    theAlignableExtras = new AlignableExtras();
  }

  // Create alignment parameter builder
  edm::LogInfo("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::init"
                            << "Creating AlignmentParameterBuilder";

  edm::ParameterSet aliParamBuildCfg =
      theParameterSet.getParameter<edm::ParameterSet>("ParameterBuilder");
  AlignmentParameterBuilder alignmentParameterBuilder(theAlignableTracker,
                                                      theAlignableExtras,
                                                      aliParamBuildCfg );

  // Fix alignables if requested
  if (stNFixAlignables_ > 0) {
    alignmentParameterBuilder.fixAlignables(stNFixAlignables_);
  }

  // Get list of alignables
  Alignables theAlignables = alignmentParameterBuilder.alignables();
  edm::LogInfo("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::beginOfJob"
                            << "got " << theAlignables.size() << " alignables";

  // Create AlignmentParameterStore
  edm::ParameterSet aliParamStoreCfg = theParameterSet.getParameter<edm::ParameterSet>("ParameterStore");
  theAlignmentParameterStore = new AlignmentParameterStore(theAlignables, aliParamStoreCfg);
  edm::LogInfo("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::beginOfJob"
                            << "AlignmentParameterStore created!";

  // Apply misalignment scenario to alignable tracker and muon if requested
  // WARNING: this assumes scenarioConfig can be passed to both muon and tracker
  if (doMisalignmentScenario_ && doTracker_) {
    edm::LogInfo("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::beginOfJob"
                              << "Applying misalignment scenario to "
                              << (doTracker_ ? "tracker" : "");
    edm::ParameterSet scenarioConfig = theParameterSet.getParameter<edm::ParameterSet>( "MisalignmentScenario" );

    TrackerScenarioBuilder scenarioBuilder(theAlignableTracker);
    scenarioBuilder.applyScenario(scenarioConfig);

  } else {
    edm::LogInfo("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::beginOfJob"
                              << "NOT applying misalignment scenario!";
  }

  // Apply simple misalignment
  const std::string sParSel(theParameterSet.getParameter<std::string>("parameterSelectorSimple"));
  simpleMisalignment(theAlignables, sParSel, stRandomShift_, stRandomRotation_, true);

  // Initialize alignment algorithm and integrated calibration and pass the latter to algorithm
  theAlignmentAlgo->initialize(setup,
                               theAlignableTracker,
                               0, // theAlignableMuon,
                               theAlignableExtras,
                               theAlignmentParameterStore);

  for (auto iCal = theCalibrations.begin(); iCal != theCalibrations.end(); ++iCal) {
    (*iCal)->beginOfJob(theAlignableTracker, 0, theAlignableExtras);
  }

  // Not all algorithms support calibrations - so do not pass empty vector
  // and throw if non-empty and not supported:
  if (!theCalibrations.empty() && !theAlignmentAlgo->addCalibrations(theCalibrations)) {
    throw cms::Exception("BadConfig") << "[TrackerAlignmentProducerForPCL::init]\n"
                      << "Configured " << theCalibrations.size() << " calibration(s) "
                      << "for algorithm not supporting it.";
  }




  /* 1) Former: AlignmentProducer::startingNewLoop(unsigned int iLoop) */

  nevent_ = 0;

  theAlignmentAlgo->startNewLoop();
  // FIXME: Should this be done in algorithm::startNewLoop()??
  for (auto iCal = theCalibrations.begin(); iCal != theCalibrations.end(); ++iCal) {
    (*iCal)->startNewLoop();
  }

  edm::LogInfo("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::init"
                            << "Now physically apply alignments to  geometry...";

  // Propagate changes to reconstruction geometry (from initialisation or iteration)
  GeometryAligner aligner;

  std::auto_ptr<Alignments>                   alignments     (theAlignableTracker->alignments());
  std::auto_ptr<AlignmentErrorsExtended>              alignmentErrors(theAlignableTracker->alignmentErrors());
  std::auto_ptr<AlignmentSurfaceDeformations> aliDeforms     (theAlignableTracker->surfaceDeformations());

  aligner.applyAlignments<TrackerGeometry>          (&(*theTracker), &(*alignments), &(*alignmentErrors), AlignTransform() ); // don't apply global a second time!
  aligner.attachSurfaceDeformations<TrackerGeometry>(&(*theTracker), &(*aliDeforms));
}

void TrackerAlignmentProducerForPCL::finish() {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);
  edm::LogInfo("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::finish";

  /* 1) Former: Status AlignmentProducer::endOfLoop(const edm::EventSetup& iSetup, unsigned int iLoop) */
  // if (0 == nevent_) {
  //   // beginOfJob is usually called by the framework in the first event of the first loop
  //   // (a hack: beginOfJob needs the EventSetup that is not well defined without an event)
  //   // and the algorithms rely on the initialisations done in beginOfJob. We cannot call
  //   // this->beginOfJob(iSetup); here either since that will access the EventSetup to get
  //   // some geometry information that is not defined either without having seen an event.
  //   edm::LogError("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::endJob"
  //                              << "Did not process any events, "
  //                              << "stop processing without terminating algorithm.";
  //   return;
  // }

  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);
  edm::LogInfo("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::endJob"
                            << "Terminating algorithm.";

  // TODO: Apparently, MP does not use the EventSetup parameter
  theAlignmentAlgo->terminate();

  // FIXME: Should this be done in algorithm::terminate(const edm::EventSetup& iSetup)??
  for (auto iCal = theCalibrations.begin(); iCal != theCalibrations.end(); ++iCal) {
    (*iCal)->endOfLoop();
  }





  /* 2) Former: void AlignmentProducer::endOfJob() */

  // if (0 == nevent_) {
  //   edm::LogError("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::finish" << "Did not process any "
  //                              << "events in last loop, do not dare to store to DB.";
  // } else {
    // Expand run ranges and make them unique
    edm::VParameterSet runRangeSelectionVPSet(theParameterSet.getParameter<edm::VParameterSet>("RunRangeSelection"));
    RunRanges uniqueRunRanges(makeNonOverlappingRunRanges(runRangeSelectionVPSet));

    // create dummy IOV
    if (uniqueRunRanges.empty()) {
      const RunRange runRange(cond::timeTypeSpecs[cond::runnumber].beginValue,
                              cond::timeTypeSpecs[cond::runnumber].endValue);
      uniqueRunRanges.push_back(runRange);
    }

    std::vector<AlgebraicVector> beamSpotParameters;

    for (RunRanges::const_iterator iRunRange = uniqueRunRanges.begin();
         iRunRange != uniqueRunRanges.end();
         ++iRunRange) {

      theAlignmentAlgo->setParametersForRunRange(*iRunRange);

      // Save alignments to database
      if (saveToDB || saveApeToDB || saveDeformationsToDB) {
        writeForRunRange((*iRunRange).first);
      }

      // Deal with extra alignables, e.g. beam spot
      if (theAlignableExtras) {
        Alignables &alis = theAlignableExtras->beamSpot();
        if (!alis.empty()) {
          BeamSpotAlignmentParameters *beamSpotAliPars = dynamic_cast<BeamSpotAlignmentParameters*>(alis[0]->alignmentParameters());
          beamSpotParameters.push_back(beamSpotAliPars->parameters());
        }
      }
    // }

    if (theAlignableExtras) {
      std::ostringstream bsOutput;

      std::vector<AlgebraicVector>::const_iterator itPar = beamSpotParameters.begin();
      for (RunRanges::const_iterator iRunRange = uniqueRunRanges.begin();
           iRunRange != uniqueRunRanges.end();
           ++iRunRange, ++itPar) {
        bsOutput << "Run range: " << (*iRunRange).first << " - " << (*iRunRange).second << "\n";
        bsOutput << "  Displacement: x=" << (*itPar)[0] << ", y=" << (*itPar)[1] << "\n";
        bsOutput << "  Slope: dx/dz=" << (*itPar)[2] << ", dy/dz=" << (*itPar)[3] << "\n";
      }

      edm::LogInfo("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::finish"
                                << "Parameters for alignable beamspot:\n"
                                << bsOutput.str();
    }

    for (auto iCal = theCalibrations.begin(); iCal != theCalibrations.end(); ++iCal) {
      (*iCal)->endOfJob();
    }
  }
}

void TrackerAlignmentProducerForPCL::createGeometries(const edm::EventSetup& setup) {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);

  //edm::ESTransientHandle<DDCompactView> cpv;
  //iSetup.get<IdealGeometryRecord>().get(cpv);

  edm::ESHandle<GeometricDet> geometricDet;
  setup.get<IdealGeometryRecord>().get(geometricDet);
  TrackerGeomBuilderFromGeometricDet trackerBuilder;

  theTracker = boost::shared_ptr<TrackerGeometry>(trackerBuilder.build(&(*geometricDet), theParameterSet));
}

void TrackerAlignmentProducerForPCL::simpleMisalignment(const Alignables& alivec, const std::string& selection,
                                                        float shift, float rot, bool local) {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);

  std::ostringstream output; // collecting output

  if (shift > 0. || rot > 0.) {
    output << "Adding random flat shift of max size " << shift
           << " and adding random flat rotation of max size " << rot <<" to ";

    std::vector<bool> commSel(0);
    if (selection != "-1") {
      AlignmentParameterSelector aSelector(0,0); // no alignable needed here...
      const std::vector<char> cSel(aSelector.convertParamSel(selection));

      if (cSel.size() < RigidBodyAlignmentParameters::N_PARAM) {
        throw cms::Exception("BadConfig")
          << "[AlignmentProducer::simpleMisalignment_]\n"
          << "Expect selection string '" << selection << "' to be at least of length "
          << RigidBodyAlignmentParameters::N_PARAM << " or to be '-1'.\n"
          << "(Most probably you have to adjust the parameter 'parameterSelectorSimple'.)";
      }

      for (std::vector<char>::const_iterator cIter = cSel.begin(); cIter != cSel.end(); ++cIter) {
        commSel.push_back(*cIter == '0' ? false : true);
      }

      output << "parameters defined by (" << selection
             << "), representing (x,y,z,alpha,beta,gamma),";

    } else {
      output << "the active parameters of each alignable,";
    }
    output << " in " << (local ? "local" : "global") << " frame.";

    for (std::vector<Alignable*>::const_iterator it = alivec.begin(); it != alivec.end(); ++it) {
      Alignable* ali = (*it);
      std::vector<bool> mysel(commSel.empty() ? ali->alignmentParameters()->selector() : commSel);

      if (std::abs(shift) > 0.00001) {
        double s0 = 0., s1 = 0., s2 = 0.;
        if (mysel[RigidBodyAlignmentParameters::dx]) s0 = shift * double(random()%1000-500)/500.;
        if (mysel[RigidBodyAlignmentParameters::dy]) s1 = shift * double(random()%1000-500)/500.;
        if (mysel[RigidBodyAlignmentParameters::dz]) s2 = shift * double(random()%1000-500)/500.;

        if (local) {
          ali->move( ali->surface().toGlobal(align::LocalVector(s0,s1,s2)) );
        } else {
          ali->move( align::GlobalVector(s0,s1,s2) );
        }
      }

      if (std::abs(rot)>0.00001) {
        align::EulerAngles r(3);
        if (mysel[RigidBodyAlignmentParameters::dalpha]) r(1)=rot*double(random()%1000-500)/500.;
        if (mysel[RigidBodyAlignmentParameters::dbeta] ) r(2)=rot*double(random()%1000-500)/500.;
        if (mysel[RigidBodyAlignmentParameters::dgamma]) r(3)=rot*double(random()%1000-500)/500.;

        const align::RotationType mrot = align::toMatrix(r);
        if (local) ali->rotateInLocalFrame(mrot);
        else       ali->rotateInGlobalFrame(mrot);
      }
    } // end loop on alignables

  } else {
    output << "No simple misalignment added!";
  }

  edm::LogInfo("Alignment") << "@SUB=TrackerAlignmentProducerForPCL::simpleMisalignment" << output.str();
}

//////////////////////////////////////////////////
// a templated method - but private, so not accessible from outside
// ==> does not have to be in header file
template<class G, class Rcd, class ErrRcd>
void TrackerAlignmentProducerForPCL::applyDB(G* geometry, const edm::EventSetup& setup,
                                             const AlignTransform& globalCoordinates) const {
    printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);
  // 'G' is the geometry class for that DB should be applied,
  // 'Rcd' is the record class for its Alignments
  // 'ErrRcd' is the record class for its AlignmentErrors
  // 'globalCoordinates' are global transformation for this geometry

  const Rcd & record = setup.get<Rcd>();
  if (checkDbAlignmentValidity_) {
    const edm::ValidityInterval& validity = record.validityInterval();
    const edm::IOVSyncValue      first    = validity.first();
    const edm::IOVSyncValue      last     = validity.last();

    if (first != edm::IOVSyncValue::beginOfTime() ||
        last  != edm::IOVSyncValue::endOfTime()) {
      throw cms::Exception("DatabaseError")
        << "@SUB=AlignmentProducer::applyDB"
        << "\nTrying to apply "
        << record.key().name()
        << " with multiple IOVs in tag.\n"
        << "Validity range is "
        << first.eventID().run() << " - " << last.eventID().run();
    }
  }

  edm::ESHandle<Alignments> alignments;
  record.get(alignments);

  edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
  setup.get<ErrRcd>().get(alignmentErrors);

  GeometryAligner aligner;
  aligner.applyAlignments<G>(geometry, &(*alignments), &(*alignmentErrors),
                 globalCoordinates);
}

//////////////////////////////////////////////////
// a templated method - but private, so not accessible from outside
// ==> does not have to be in header file
template<class G, class DeformationRcd>
void TrackerAlignmentProducerForPCL::applyDB(G* geometry, const edm::EventSetup& setup) const {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);
  // 'G' is the geometry class for that DB should be applied,
  // 'DeformationRcd' is the record class for its surface deformations

  const DeformationRcd & record = setup.get<DeformationRcd>();
  if (checkDbAlignmentValidity_) {
    const edm::ValidityInterval & validity = record.validityInterval();
    const edm::IOVSyncValue first = validity.first();
    const edm::IOVSyncValue last = validity.last();

    if (first != edm::IOVSyncValue::beginOfTime() ||
        last  != edm::IOVSyncValue::endOfTime()) {
      throw cms::Exception("DatabaseError")
        << "@SUB=AlignmentProducer::applyDB"
        << "\nTrying to apply "
        << record.key().name()
        << " with multiple IOVs in tag.\n"
        << "Validity range is "
        << first.eventID().run() << " - " << last.eventID().run();
    }
  }

  edm::ESHandle<AlignmentSurfaceDeformations> surfaceDeformations;
  record.get(surfaceDeformations);

  GeometryAligner aligner;
  aligner.attachSurfaceDeformations<G>(geometry, &(*surfaceDeformations));
}

void TrackerAlignmentProducerForPCL::writeForRunRange(cond::Time_t time) {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);
  // will be 'removed' from constants
  const AlignTransform* trackerGlobal = 0;

  // i.e. applied before in applyDB
  if (globalPositions) {
    trackerGlobal = &align::DetectorGlobalPosition(*globalPositions,
                                                    DetId(DetId::Tracker));
    }

    Alignments*      alignments      = theAlignableTracker->alignments();
    AlignmentErrorsExtended* alignmentErrors = theAlignableTracker->alignmentErrors();

    writeDB(alignments, "TrackerAlignmentRcd",
            alignmentErrors, "TrackerAlignmentErrorExtendedRcd",
            trackerGlobal, time);


  // Save surface deformations to database
  if (saveDeformationsToDB) {
    AlignmentSurfaceDeformations* alignmentSurfaceDeformations = theAlignableTracker->surfaceDeformations();
    writeDB(alignmentSurfaceDeformations, "TrackerSurfaceDeformationRcd", time);
  }
}

void TrackerAlignmentProducerForPCL::writeDB(Alignments* alignments,
                                             const std::string& alignRcd,
                                             AlignmentErrorsExtended* alignmentErrors,
                                             const std::string& errRcd,
                                             const AlignTransform* globalCoordinates,
                                             cond::Time_t time) const {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);

  Alignments*      tempAlignments      = alignments;
  AlignmentErrorsExtended* tempAlignmentErrors = alignmentErrors;

  // Call service
  edm::Service<cond::service::PoolDBOutputService> poolDb;
  // Die if not available
  if (!poolDb.isAvailable()) {
    // promised to take over ownership...
    delete tempAlignments;
    delete tempAlignmentErrors; // dito
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
  }

  if (globalCoordinates && // happens only if (applyDbAlignment_ == true)
      globalCoordinates->transform() != AlignTransform::Transform::Identity) {

    tempAlignments      = new Alignments();            // temporary storage for
    tempAlignmentErrors = new AlignmentErrorsExtended();  // final alignments and errors

    GeometryAligner aligner;
    aligner.removeGlobalTransform(alignments, alignmentErrors,
                                  *globalCoordinates,
                                  tempAlignments, tempAlignmentErrors);

    delete alignments;       // have to delete original alignments
    delete alignmentErrors;  // same thing for the errors

    edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::writeDB"
                              << "globalCoordinates removed from alignments (" << alignRcd
                              << ") and errors (" << alignRcd << ").";
  }

  if (saveToDB) {
    edm::LogInfo("Alignment") << "Writing Alignments for run " << time
                              << " to " << alignRcd << ".";
    poolDb->writeOne<Alignments>(tempAlignments, time, alignRcd);

  } else {
    // poolDb->writeOne(..) takes over 'alignments' ownership,...
    delete tempAlignments; // ...otherwise we have to delete, as promised!
  }

  if (saveApeToDB) {
    edm::LogInfo("Alignment") << "Writing AlignmentErrors for run " << time
                              << " to " << errRcd << ".";
    poolDb->writeOne<AlignmentErrorsExtended>(tempAlignmentErrors, time, errRcd);

  } else {
    // poolDb->writeOne(..) takes over 'alignmentErrors' ownership,...
    delete tempAlignmentErrors; // ...otherwise we have to delete, as promised!
  }
}

void TrackerAlignmentProducerForPCL::writeDB(AlignmentSurfaceDeformations* alignmentSurfaceDeformations,
                                             const std::string& surfaceDeformationRcd,
                                             cond::Time_t time) const {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);

  // Call service
  edm::Service<cond::service::PoolDBOutputService> poolDb;
  // Die if not available
  if (!poolDb.isAvailable()) {
    delete alignmentSurfaceDeformations; // promised to take over ownership...
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
  }

  if (saveDeformationsToDB) {
    edm::LogInfo("Alignment") << "Writing AlignmentSurfaceDeformations for run " << time
                              << " to " << surfaceDeformationRcd  << ".";
    poolDb->writeOne<AlignmentSurfaceDeformations>(alignmentSurfaceDeformations, time,
                                                   surfaceDeformationRcd);
  } else {
    // poolDb->writeOne(..) takes over 'surfaceDeformation' ownership,...
    delete alignmentSurfaceDeformations; // ...otherwise we have to delete, as promised!
  }
}

TrackerAlignmentProducerForPCL::RunRanges TrackerAlignmentProducerForPCL::makeNonOverlappingRunRanges(
    const edm::VParameterSet& RunRangeSelectionVPSet) {
  printf("(TrackerAlignmentProducerForPCL) function %s in %s was called\n", __FUNCTION__, __FILE__);

  static bool oldRunRangeSelectionWarning = false;

  const RunNumber beginValue = cond::timeTypeSpecs[cond::runnumber].beginValue;
  const RunNumber endValue   = cond::timeTypeSpecs[cond::runnumber].endValue;

  RunRanges uniqueRunRanges;
  if (!RunRangeSelectionVPSet.empty()) {

    std::map<RunNumber,RunNumber> uniqueFirstRunNumbers;

    for (std::vector<edm::ParameterSet>::const_iterator ipset = RunRangeSelectionVPSet.begin();
         ipset != RunRangeSelectionVPSet.end(); ++ipset) {
      const std::vector<std::string> RunRangeStrings = (*ipset).getParameter<std::vector<std::string> >("RunRanges");

      for (std::vector<std::string>::const_iterator irange = RunRangeStrings.begin();
           irange != RunRangeStrings.end(); ++irange) {

        if ((*irange).find(':') == std::string::npos) {
          RunNumber first = beginValue;
          long int temp = strtol((*irange).c_str(), 0, 0);
          if (temp != -1) first = temp;
          uniqueFirstRunNumbers[first] = first;

        } else {
          if (!oldRunRangeSelectionWarning) {
            edm::LogWarning("BadConfig") << "@SUB=AlignmentProducer::makeNonOverlappingRunRanges"
                     << "Config file contains old format for 'RunRangeSelection'. Only the start run\n"
                     << "number is used internally. The number of the last run is ignored and can be\n"
                     << "safely removed from the config file.\n";
            oldRunRangeSelectionWarning = true;
          }

          std::vector<std::string> tokens = edm::tokenize(*irange, ":");
          long int temp;
          RunNumber first = beginValue;
          temp = strtol(tokens[0].c_str(), 0, 0);
          if (temp != -1) first = temp;
          uniqueFirstRunNumbers[first] = first;
        }
      }
    }

    for (std::map<RunNumber,RunNumber>::iterator iFirst = uniqueFirstRunNumbers.begin();
         iFirst!=uniqueFirstRunNumbers.end(); ++iFirst) {
      uniqueRunRanges.push_back(std::pair<RunNumber,RunNumber>((*iFirst).first, endValue));
    }

    for (unsigned int i = 0; i < uniqueRunRanges.size()-1; ++i) {
      uniqueRunRanges[i].second = uniqueRunRanges[i+1].first - 1;
    }

  } else {
    uniqueRunRanges.push_back(std::pair<RunNumber,RunNumber>(beginValue, endValue));
  }

  return uniqueRunRanges;
}



void TrackerAlignmentProducerForPCL::addSurveyInfo(Alignable* ali) {
  const std::vector<Alignable*>& comp = ali->components();
  for (unsigned int i = 0; i < comp.size(); ++i) {
    addSurveyInfo(comp[i]);
  }

  const SurveyError& error = theSurveyErrors->m_surveyErrors[theSurveyIndex];

  if (ali->id()                != error.rawId() ||
      ali->alignableObjectId() != error.structureType()) {
    throw cms::Exception("DatabaseError") << "Error reading survey info from DB. "
                                             "Mismatched id!";
  }

  const CLHEP::Hep3Vector&  pos = theSurveyValues->m_align[theSurveyIndex].translation();
  const CLHEP::HepRotation& rot = theSurveyValues->m_align[theSurveyIndex].rotation();

  AlignableSurface surf(align::PositionType(pos.x(),  pos.y(),  pos.z()),
                        align::RotationType(rot.xx(), rot.xy(), rot.xz(),
                        rot.yx(), rot.yy(), rot.yz(),
                        rot.zx(), rot.zy(), rot.zz() ) );
  surf.setWidth (ali->surface().width());
  surf.setLength(ali->surface().length());

  ali->setSurvey(new SurveyDet(surf, error.matrix()));

  ++theSurveyIndex;
}

void TrackerAlignmentProducerForPCL::readInSurveyRcds(const edm::EventSetup& setup) {
  if (useSurvey_) {
    bool tkSurveyBool    = watchTkSurveyRcd_.check(setup);
    bool tkSurveyErrBool = watchTkSurveyErrRcd_.check(setup);
    edm::LogInfo("Alignment") << "watcher tksurveyrcd: "    << tkSurveyBool;
    edm::LogInfo("Alignment") << "watcher tksurveyerrrcd: " << tkSurveyErrBool;

    if (tkSurveyBool || tkSurveyErrBool) {
      edm::LogInfo("Alignment") << "ADDING THE SURVEY INFORMATION";
      edm::ESHandle<Alignments> surveys;
      edm::ESHandle<SurveyErrors> surveyErrors;

      setup.get<TrackerSurveyRcd>().get(surveys);
      setup.get<TrackerSurveyErrorExtendedRcd>().get(surveyErrors);

      theSurveyIndex  = 0;
      theSurveyValues = &*surveys;
      theSurveyErrors = &*surveyErrors;

      addSurveyInfo(theAlignableTracker);
    }
  }
}

// define this as a plugin
DEFINE_FWK_MODULE(TrackerAlignmentProducerForPCL);
