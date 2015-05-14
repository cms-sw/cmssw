/// \file AlignmentProducer.cc
///
///  \author    : Frederic Ronga
///  Revision   : $Revision: 1.68 $
///  last update: $Date: 2012/08/10 09:25:23 $
///  by         : $Author: flucke $

#include "AlignmentProducer.h"
#include "FWCore/Framework/interface/LooperFactory.h" 
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
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/Utilities/interface/Parse.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurveyErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTSurveyErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCSurveyErrorExtendedRcd.h"
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
#include "Alignment/MuonAlignment/interface/MuonScenarioBuilder.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/BeamSpotAlignmentParameters.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/IntegratedCalibrationPluginFactory.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"

//_____________________________________________________________________________
AlignmentProducer::AlignmentProducer(const edm::ParameterSet& iConfig) :
  theAlignmentAlgo(0), theAlignmentParameterStore(0),
  theAlignableExtras(0), theAlignableTracker(0), theAlignableMuon(0), 
  globalPositions_(0),
  nevent_(0), theParameterSet(iConfig),
  theMaxLoops( iConfig.getUntrackedParameter<unsigned int>("maxLoops") ),
  stNFixAlignables_(iConfig.getParameter<int>("nFixAlignables") ),
  stRandomShift_(iConfig.getParameter<double>("randomShift")),
  stRandomRotation_(iConfig.getParameter<double>("randomRotation")),
  applyDbAlignment_( iConfig.getUntrackedParameter<bool>("applyDbAlignment")),
  checkDbAlignmentValidity_( iConfig.getUntrackedParameter<bool>("checkDbAlignmentValidity")),
  doMisalignmentScenario_(iConfig.getParameter<bool>("doMisalignmentScenario")),
  saveToDB_(iConfig.getParameter<bool>("saveToDB")),
  saveApeToDB_(iConfig.getParameter<bool>("saveApeToDB")),
  saveDeformationsToDB_(iConfig.getParameter<bool>("saveDeformationsToDB")),
  doTracker_( iConfig.getUntrackedParameter<bool>("doTracker") ),
  doMuon_( iConfig.getUntrackedParameter<bool>("doMuon") ),
  useExtras_( iConfig.getUntrackedParameter<bool>("useExtras") ),
  useSurvey_( iConfig.getParameter<bool>("useSurvey") ),
  tjTkAssociationMapTag_(iConfig.getParameter<edm::InputTag>("tjTkAssociationMapTag")),
  beamSpotTag_(iConfig.getParameter<edm::InputTag>("beamSpotTag")),
  tkLasBeamTag_(iConfig.getParameter<edm::InputTag>("tkLasBeamTag")),
  clusterValueMapTag_(iConfig.getParameter<edm::InputTag>("hitPrescaleMapTag"))
{
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::AlignmentProducer";

  // Tell the framework what data is being produced
  if (doTracker_) {
     setWhatProduced(this, &AlignmentProducer::produceTracker);
  }
  if (doMuon_) {
     setWhatProduced(this, &AlignmentProducer::produceDT);
     setWhatProduced(this, &AlignmentProducer::produceCSC);
  }

  // Create the alignment algorithm
  edm::ParameterSet algoConfig = iConfig.getParameter<edm::ParameterSet>( "algoConfig" );
  edm::VParameterSet iovSelection = iConfig.getParameter<edm::VParameterSet>( "RunRangeSelection" );
  algoConfig.addUntrackedParameter<edm::VParameterSet>( "RunRangeSelection", iovSelection );
  std::string algoName = algoConfig.getParameter<std::string>( "algoName" );
  theAlignmentAlgo = AlignmentAlgorithmPluginFactory::get( )->create( algoName, algoConfig  );

  // Check if found
  if ( !theAlignmentAlgo )
	throw cms::Exception("BadConfig") << "Couldn't find algorithm called " << algoName;

  // Now create monitors:
  edm::ParameterSet monitorConfig = iConfig.getParameter<edm::ParameterSet>( "monitorConfig" );
  std::vector<std::string> monitors = monitorConfig.getUntrackedParameter<std::vector<std::string> >( "monitors" );
  for (std::vector<std::string>::const_iterator miter = monitors.begin();  miter != monitors.end();  ++miter) {
    AlignmentMonitorBase* newMonitor = AlignmentMonitorPluginFactory::get()->create(*miter, monitorConfig.getUntrackedParameter<edm::ParameterSet>(*miter));

    if (!newMonitor) throw cms::Exception("BadConfig") << "Couldn't find monitor named " << *miter;

    theMonitors.push_back(newMonitor);
  }

  // Finally create integrated calibrations:
  edm::VParameterSet calibrations = iConfig.getParameter<edm::VParameterSet>("calibrations");
  for (auto iCalib = calibrations.begin(); iCalib != calibrations.end(); ++iCalib) {
    const std::string name(iCalib->getParameter<std::string>("calibrationName"));
    theCalibrations.push_back(IntegratedCalibrationPluginFactory::get()->create(name, *iCalib));
    // exception comes from line before: if (!theCalibrations.back()) throw cms::Exception(..) << ..;
  }

}


//_____________________________________________________________________________
// Delete new objects
AlignmentProducer::~AlignmentProducer()
{
  delete theAlignmentAlgo;

  // Delete monitors as well??

  for (auto iCal = theCalibrations.begin(); iCal != theCalibrations.end(); ++iCal) {
    delete *iCal; // delete integrated calibration pointed to by (*iCal)
  }

  delete theAlignmentParameterStore;
  delete theAlignableExtras;
  delete theAlignableTracker;
  delete theAlignableMuon;

  delete globalPositions_;
}


//_____________________________________________________________________________
// Produce tracker geometry
boost::shared_ptr<TrackerGeometry>
AlignmentProducer::produceTracker( const TrackerDigiGeometryRecord& iRecord )
{
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::produceTracker";
  return theTracker;
}

//_____________________________________________________________________________
// Produce muonDT geometry
boost::shared_ptr<DTGeometry>
AlignmentProducer::produceDT( const MuonGeometryRecord& iRecord )
{
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::produceDT";
  return theMuonDT;
}

//_____________________________________________________________________________
// Produce muonCSC geometry
boost::shared_ptr<CSCGeometry>
AlignmentProducer::produceCSC( const MuonGeometryRecord& iRecord )
{
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::produceCSC";
  return theMuonCSC;  
}


//_____________________________________________________________________________
// Initialize algorithm
void AlignmentProducer::beginOfJob( const edm::EventSetup& iSetup )
{
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob";

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // Create the geometries from the ideal geometries (first time only)
  this->createGeometries_( iSetup );
  
  // Retrieve and apply alignments, if requested (requires DB setup)
  if ( applyDbAlignment_ ) {
    // we need GlobalPositionRcd - and have to keep track for later removal
    // before writing again to DB...
    edm::ESHandle<Alignments> globalPositionRcd;
    iSetup.get<GlobalPositionRcd>().get(globalPositionRcd);
    globalPositions_ = new Alignments(*globalPositionRcd);

    if ( doTracker_ ) {     // apply to tracker
      this->applyDB<TrackerGeometry,TrackerAlignmentRcd,TrackerAlignmentErrorExtendedRcd>
	(&(*theTracker), iSetup,  
	 align::DetectorGlobalPosition(*globalPositions_, DetId(DetId::Tracker)));
      this->applyDB<TrackerGeometry,TrackerSurfaceDeformationRcd>(&(*theTracker), iSetup);
    }
    
    if ( doMuon_ ) { // apply to tracker
      this->applyDB<DTGeometry,DTAlignmentRcd,DTAlignmentErrorExtendedRcd>
	(&(*theMuonDT), iSetup,
	 align::DetectorGlobalPosition(*globalPositions_, DetId(DetId::Muon)));
      this->applyDB<CSCGeometry,CSCAlignmentRcd,CSCAlignmentErrorExtendedRcd>
	(&(*theMuonCSC), iSetup,
	 align::DetectorGlobalPosition(*globalPositions_, DetId(DetId::Muon)));
    }
  }

  // Create alignable tracker and muon 
  if (doTracker_) {
    theAlignableTracker = new AlignableTracker( &(*theTracker), tTopo );
  }

  if (doMuon_) {
     theAlignableMuon = new AlignableMuon( &(*theMuonDT), &(*theMuonCSC) );
  }

  if (useExtras_) {
    theAlignableExtras = new AlignableExtras();
  }

  // Create alignment parameter builder
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob" 
                            << "Creating AlignmentParameterBuilder";
  edm::ParameterSet aliParamBuildCfg = 
    theParameterSet.getParameter<edm::ParameterSet>("ParameterBuilder");
  AlignmentParameterBuilder alignmentParameterBuilder(theAlignableTracker,
                                                      theAlignableMuon,
                                                      theAlignableExtras,
						      aliParamBuildCfg );
  // Fix alignables if requested
  if (stNFixAlignables_>0) alignmentParameterBuilder.fixAlignables(stNFixAlignables_);

  // Get list of alignables
  Alignables theAlignables = alignmentParameterBuilder.alignables();
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob" 
                            << "got " << theAlignables.size() << " alignables";

  // Create AlignmentParameterStore 
  edm::ParameterSet aliParamStoreCfg = 
    theParameterSet.getParameter<edm::ParameterSet>("ParameterStore");
  theAlignmentParameterStore = new AlignmentParameterStore(theAlignables, aliParamStoreCfg);
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob" 
                            << "AlignmentParameterStore created!";

  // Apply misalignment scenario to alignable tracker and muon if requested
  // WARNING: this assumes scenarioConfig can be passed to both muon and tracker
  if (doMisalignmentScenario_ && (doTracker_ || doMuon_)) {
    edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob" 
                              << "Applying misalignment scenario to "
                              << (doTracker_ ? "tracker" : "")
                              << (doMuon_    ? (doTracker_ ? " and muon" : "muon") : ".");
    edm::ParameterSet scenarioConfig 
      = theParameterSet.getParameter<edm::ParameterSet>( "MisalignmentScenario" );
    if (doTracker_) {
      TrackerScenarioBuilder scenarioBuilder( theAlignableTracker );
      scenarioBuilder.applyScenario( scenarioConfig );
    }
    if (doMuon_) {
      MuonScenarioBuilder muonScenarioBuilder( theAlignableMuon );
      muonScenarioBuilder.applyScenario( scenarioConfig );
    }
  } else {
    edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob" 
                              << "NOT applying misalignment scenario!";
  }

  // Apply simple misalignment
  const std::string sParSel(theParameterSet.getParameter<std::string>("parameterSelectorSimple"));
  this->simpleMisalignment_(theAlignables, sParSel, stRandomShift_, stRandomRotation_, true);

  // Initialize alignment algorithm and integrated calibration and pass the latter to algorithm
  theAlignmentAlgo->initialize( iSetup, 
				theAlignableTracker, theAlignableMuon, theAlignableExtras,
				theAlignmentParameterStore );
  for (auto iCal = theCalibrations.begin(); iCal != theCalibrations.end(); ++iCal) {
    (*iCal)->beginOfJob(theAlignableTracker, theAlignableMuon, theAlignableExtras);
  }
  // Not all algorithms support calibrations - so do not pass empty vector
  // and throw if non-empty and not supported:
  if (!theCalibrations.empty() && !theAlignmentAlgo->addCalibrations(theCalibrations)) {
    throw cms::Exception("BadConfig") << "[AlignmentProducer::beginOfJob]\n"
				      << "Configured " << theCalibrations.size() << " calibration(s) "
				      << "for algorithm not supporting it.";
  }

  for (std::vector<AlignmentMonitorBase*>::const_iterator monitor = theMonitors.begin();
       monitor != theMonitors.end();  ++monitor) {
     (*monitor)->beginOfJob(theAlignableTracker, theAlignableMuon, theAlignmentParameterStore);
  }
}

//_____________________________________________________________________________
// Terminate algorithm
void AlignmentProducer::endOfJob()
{
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::endOfJob";

  for (std::vector<AlignmentMonitorBase*>::const_iterator monitor = theMonitors.begin();  monitor != theMonitors.end();  ++monitor) {
     (*monitor)->endOfJob();
  }

  if (0 == nevent_) {
    edm::LogError("Alignment") << "@SUB=AlignmentProducer::endOfJob" << "Did not process any "
                               << "events in last loop, do not dare to store to DB.";
  } else {
    
    // Expand run ranges and make them unique
    edm::VParameterSet runRangeSelectionVPSet(theParameterSet.getParameter<edm::VParameterSet>("RunRangeSelection"));
    RunRanges uniqueRunRanges(this->makeNonOverlappingRunRanges(runRangeSelectionVPSet));
    if (uniqueRunRanges.empty()) { // create dummy IOV
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
      if (saveToDB_ || saveApeToDB_ || saveDeformationsToDB_)
        this->writeForRunRange((*iRunRange).first);
      
      // Deal with extra alignables, e.g. beam spot
      if (theAlignableExtras) {
	Alignables &alis = theAlignableExtras->beamSpot();
	if (!alis.empty()) {
	  BeamSpotAlignmentParameters *beamSpotAliPars = dynamic_cast<BeamSpotAlignmentParameters*>(alis[0]->alignmentParameters());
	  beamSpotParameters.push_back(beamSpotAliPars->parameters());
	}
      }
    }
    
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
      
      edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::endOfJob"
				<< "Parameters for alignable beamspot:\n"
				<< bsOutput.str();
    }

    for (auto iCal = theCalibrations.begin(); iCal != theCalibrations.end(); ++iCal) {
      (*iCal)->endOfJob();
    }

  }
}

//_____________________________________________________________________________
// Called at beginning of loop
void AlignmentProducer::startingNewLoop(unsigned int iLoop )
{
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::startingNewLoop" 
                            << "Starting loop number " << iLoop;

  nevent_ = 0;

  theAlignmentAlgo->startNewLoop();
  // FIXME: Should this be done in algorithm::startNewLoop()??
  for (auto iCal = theCalibrations.begin(); iCal != theCalibrations.end(); ++iCal) {
    (*iCal)->startNewLoop();
  }

  for (std::vector<AlignmentMonitorBase*>::const_iterator monitor = theMonitors.begin();  monitor != theMonitors.end();  ++monitor) {
     (*monitor)->startingNewLoop();
  }

  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::startingNewLoop" 
                            << "Now physically apply alignments to  geometry...";


  // Propagate changes to reconstruction geometry (from initialisation or iteration)
  GeometryAligner aligner;
  if ( doTracker_ ) {
    std::auto_ptr<Alignments> alignments(theAlignableTracker->alignments());
    std::auto_ptr<AlignmentErrorsExtended> alignmentErrors(theAlignableTracker->alignmentErrors());
    aligner.applyAlignments<TrackerGeometry>( &(*theTracker),&(*alignments),&(*alignmentErrors), AlignTransform() ); // don't apply global a second time!
    std::auto_ptr<AlignmentSurfaceDeformations> aliDeforms(theAlignableTracker->surfaceDeformations());
    aligner.attachSurfaceDeformations<TrackerGeometry>(&(*theTracker), &(*aliDeforms));

  }
  if ( doMuon_ ) {
    std::auto_ptr<Alignments>      dtAlignments(       theAlignableMuon->dtAlignments());
    std::auto_ptr<AlignmentErrorsExtended> dtAlignmentErrorsExtended(  theAlignableMuon->dtAlignmentErrorsExtended());
    std::auto_ptr<Alignments>      cscAlignments(      theAlignableMuon->cscAlignments());
    std::auto_ptr<AlignmentErrorsExtended> cscAlignmentErrorsExtended( theAlignableMuon->cscAlignmentErrorsExtended());

    aligner.applyAlignments<DTGeometry>( &(*theMuonDT), &(*dtAlignments), &(*dtAlignmentErrorsExtended), AlignTransform() ); // don't apply global a second time!
    aligner.applyAlignments<CSCGeometry>( &(*theMuonCSC), &(*cscAlignments), &(*cscAlignmentErrorsExtended), AlignTransform() ); // nope!
  }
}


//_____________________________________________________________________________
// Called at end of loop
edm::EDLooper::Status 
AlignmentProducer::endOfLoop(const edm::EventSetup& iSetup, unsigned int iLoop)
{

  if (0 == nevent_) {
    // beginOfJob is usually called by the framework in the first event of the first loop
    // (a hack: beginOfJob needs the EventSetup that is not well defined without an event)
    // and the algorithms rely on the initialisations done in beginOfJob. We cannot call 
    // this->beginOfJob(iSetup); here either since that will access the EventSetup to get
    // some geometry information that is not defined either without having seen an event.
    edm::LogError("Alignment") << "@SUB=AlignmentProducer::endOfLoop" 
                               << "Did not process any events in loop " << iLoop
                               << ", stop processing without terminating algorithm.";
    return kStop;
  }

  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::endOfLoop" 
                            << "Ending loop " << iLoop << ", terminating algorithm.";

  theAlignmentAlgo->terminate(iSetup);
  // FIXME: Should this be done in algorithm::terminate(const edm::EventSetup& iSetup)??
  for (auto iCal = theCalibrations.begin(); iCal != theCalibrations.end(); ++iCal) {
    (*iCal)->endOfLoop();
  }

  for (std::vector<AlignmentMonitorBase*>::const_iterator monitor = theMonitors.begin();  monitor != theMonitors.end();  ++monitor) {
     (*monitor)->endOfLoop(iSetup);
  }

  if ( iLoop == theMaxLoops-1 || iLoop >= theMaxLoops ) return kStop;
  else return kContinue;
}

//_____________________________________________________________________________
// Called at each event
edm::EDLooper::Status 
AlignmentProducer::duringLoop( const edm::Event& event, 
			       const edm::EventSetup& setup )
{
  ++nevent_;

  // reading in survey records
  this->readInSurveyRcds(setup);
	
  // Printout event number
  for ( int i=10; i<10000000; i*=10 )
    if ( nevent_<10*i && (nevent_%i)==0 )
      edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::duringLoop" 
                                << "Events processed: " << nevent_;
  
  // Retrieve trajectories and tracks from the event
  // -> merely skip if collection is empty
  edm::Handle<TrajTrackAssociationCollection> m_TrajTracksMap;
  if (event.getByLabel(tjTkAssociationMapTag_, m_TrajTracksMap)) {
    
    // Form pairs of trajectories and tracks
    ConstTrajTrackPairCollection trajTracks;
    for ( TrajTrackAssociationCollection::const_iterator iPair = m_TrajTracksMap->begin();
          iPair != m_TrajTracksMap->end(); ++iPair) {
      trajTracks.push_back( ConstTrajTrackPair( &(*(*iPair).key), &(*(*iPair).val) ) );
    }
    edm::Handle<reco::BeamSpot> beamSpot;
    event.getByLabel(beamSpotTag_, beamSpot);

    if (nevent_==1 && theAlignableExtras) {
      edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::duringLoop"
				<< "initializing AlignableBeamSpot" << std::endl;
      theAlignableExtras->initializeBeamSpot(beamSpot->x0(), beamSpot->y0(), beamSpot->z0(),
					     beamSpot->dxdz(), beamSpot->dydz());
    }

    // Run the alignment algorithm with its input
    const AliClusterValueMap *clusterValueMapPtr = 0;
    if(clusterValueMapTag_.encode().size()){//check that the input tag is not empty
      edm::Handle<AliClusterValueMap> clusterValueMap;
      event.getByLabel(clusterValueMapTag_, clusterValueMap);
      clusterValueMapPtr = &(*clusterValueMap);
    }

    const AlignmentAlgorithmBase::EventInfo eventInfo(event.id(), trajTracks, *beamSpot,
						      clusterValueMapPtr);
    theAlignmentAlgo->run(setup, eventInfo);


    for (std::vector<AlignmentMonitorBase*>::const_iterator monitor = theMonitors.begin();
	 monitor != theMonitors.end();  ++monitor) {
      (*monitor)->duringLoop(event, setup, trajTracks); // forward eventInfo?
    }
  } else {
    edm::LogError("Alignment") << "@SUB=AlignmentProducer::duringLoop" 
			       << "No track collection found: skipping event";
  }
  

  return kContinue;
}

// ----------------------------------------------------------------------------
void AlignmentProducer::beginRun(const edm::Run &run, const edm::EventSetup &setup)
{
  theAlignmentAlgo->beginRun(setup); // do not forward edm::Run...
}

// ----------------------------------------------------------------------------
void AlignmentProducer::endRun(const edm::Run &run, const edm::EventSetup &setup)
{
  // call with or without las beam info...
  typedef AlignmentAlgorithmBase::EndRunInfo EndRunInfo;
  if (tkLasBeamTag_.encode().size()) { // non-empty InputTag
    edm::Handle<TkFittedLasBeamCollection> lasBeams;
    edm::Handle<TsosVectorCollection> tsoses;
    run.getByLabel(tkLasBeamTag_, lasBeams);
    run.getByLabel(tkLasBeamTag_, tsoses);
    
    theAlignmentAlgo->endRun(EndRunInfo(run.id(), &(*lasBeams), &(*tsoses)), setup);
  } else {
    edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::endRun"
			      << "No Tk LAS beams to forward to algorithm.";
    theAlignmentAlgo->endRun(EndRunInfo(run.id(), 0, 0), setup);
  }
}

// ----------------------------------------------------------------------------
void AlignmentProducer::beginLuminosityBlock(const edm::LuminosityBlock &lumiBlock,
				    const edm::EventSetup &setup)
{
  theAlignmentAlgo->beginLuminosityBlock(setup); // do not forward edm::LuminosityBlock
}

// ----------------------------------------------------------------------------
void AlignmentProducer::endLuminosityBlock(const edm::LuminosityBlock &lumiBlock,
				  const edm::EventSetup &setup)
{
  theAlignmentAlgo->endLuminosityBlock(setup); // do not forward edm::LuminosityBlock
}

// ----------------------------------------------------------------------------

void AlignmentProducer::simpleMisalignment_(const Alignables &alivec, const std::string &selection, 
                                            float shift, float rot, bool local)
{

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
      Alignable* ali=(*it);
      std::vector<bool> mysel(commSel.empty() ? ali->alignmentParameters()->selector() : commSel);
      
      if (std::abs(shift)>0.00001) {
	double s0 = 0., s1 = 0., s2 = 0.;
        if (mysel[RigidBodyAlignmentParameters::dx]) s0 = shift * double(random()%1000-500)/500.;
        if (mysel[RigidBodyAlignmentParameters::dy]) s1 = shift * double(random()%1000-500)/500.;
        if (mysel[RigidBodyAlignmentParameters::dz]) s2 = shift * double(random()%1000-500)/500.;
        
        if (local) ali->move( ali->surface().toGlobal(align::LocalVector(s0,s1,s2)) );
	else       ali->move( align::GlobalVector(s0,s1,s2) );

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
  edm::LogInfo("Alignment")  << "@SUB=AlignmentProducer::simpleMisalignment_" << output.str();
}


//__________________________________________________________________________________________________
void AlignmentProducer::createGeometries_( const edm::EventSetup& iSetup )
{
   edm::ESTransientHandle<DDCompactView> cpv;
   iSetup.get<IdealGeometryRecord>().get( cpv );

   if (doTracker_) {
     edm::ESHandle<GeometricDet> geometricDet;
     iSetup.get<IdealGeometryRecord>().get( geometricDet );
     TrackerGeomBuilderFromGeometricDet trackerBuilder;
     theTracker = boost::shared_ptr<TrackerGeometry>( trackerBuilder.build(&(*geometricDet), theParameterSet ));
   }

   if (doMuon_) {
     edm::ESHandle<MuonDDDConstants> mdc;
     iSetup.get<MuonNumberingRecord>().get(mdc);
     DTGeometryBuilderFromDDD DTGeometryBuilder;
     CSCGeometryBuilderFromDDD CSCGeometryBuilder;
     theMuonDT = boost::shared_ptr<DTGeometry>(new DTGeometry );
     DTGeometryBuilder.build( theMuonDT, &(*cpv), *mdc);
     theMuonCSC = boost::shared_ptr<CSCGeometry>( new CSCGeometry );
     CSCGeometryBuilder.build( theMuonCSC, &(*cpv), *mdc );
   }
}

void AlignmentProducer::addSurveyInfo_(Alignable* ali)
{
  const std::vector<Alignable*>& comp = ali->components();

  unsigned int nComp = comp.size();

  for (unsigned int i = 0; i < nComp; ++i) addSurveyInfo_(comp[i]);

  const SurveyError& error = theSurveyErrors->m_surveyErrors[theSurveyIndex];

  if ( ali->id() != error.rawId() ||
       ali->alignableObjectId() != error.structureType() )
  {
    throw cms::Exception("DatabaseError")
      << "Error reading survey info from DB. Mismatched id!";
  }

  const CLHEP::Hep3Vector&  pos = theSurveyValues->m_align[theSurveyIndex].translation();
  const CLHEP::HepRotation& rot = theSurveyValues->m_align[theSurveyIndex].rotation();

  AlignableSurface surf( align::PositionType( pos.x(), pos.y(), pos.z() ),
			 align::RotationType( rot.xx(), rot.xy(), rot.xz(),
					      rot.yx(), rot.yy(), rot.yz(),
					      rot.zx(), rot.zy(), rot.zz() ) );

  surf.setWidth( ali->surface().width() );
  surf.setLength( ali->surface().length() );

  ali->setSurvey( new SurveyDet( surf, error.matrix() ) );

  ++theSurveyIndex;
}

void AlignmentProducer::readInSurveyRcds( const edm::EventSetup& iSetup ){
	
  // Get Survey Rcds and add Survey Info
  if ( doTracker_ && useSurvey_ ){
    bool tkSurveyBool = watchTkSurveyRcd_.check(iSetup);
    bool tkSurveyErrBool = watchTkSurveyErrRcd_.check(iSetup);
    edm::LogInfo("Alignment") << "watcher tksurveyrcd: " << tkSurveyBool;
    edm::LogInfo("Alignment") << "watcher tksurveyerrrcd: " << tkSurveyErrBool;
    if ( tkSurveyBool || tkSurveyErrBool){
      
      edm::LogInfo("Alignment") << "ADDING THE SURVEY INFORMATION";
      edm::ESHandle<Alignments> surveys;
      edm::ESHandle<SurveyErrors> surveyErrors;
      
      iSetup.get<TrackerSurveyRcd>().get(surveys);
      iSetup.get<TrackerSurveyErrorExtendedRcd>().get(surveyErrors);
      
      theSurveyIndex  = 0;
      theSurveyValues = &*surveys;
      theSurveyErrors = &*surveyErrors;
      addSurveyInfo_(theAlignableTracker);
    }
  }
  
  if ( doMuon_ && useSurvey_) {
    bool DTSurveyBool = watchTkSurveyRcd_.check(iSetup);
    bool DTSurveyErrBool = watchTkSurveyErrRcd_.check(iSetup);
    bool CSCSurveyBool = watchTkSurveyRcd_.check(iSetup);
    bool CSCSurveyErrBool = watchTkSurveyErrRcd_.check(iSetup);
    
    if ( DTSurveyBool || DTSurveyErrBool || CSCSurveyBool || CSCSurveyErrBool ){
      edm::ESHandle<Alignments> dtSurveys;
      edm::ESHandle<SurveyErrors> dtSurveyErrors;
      edm::ESHandle<Alignments> cscSurveys;
      edm::ESHandle<SurveyErrors> cscSurveyErrors;
      
      iSetup.get<DTSurveyRcd>().get(dtSurveys);
      iSetup.get<DTSurveyErrorExtendedRcd>().get(dtSurveyErrors);
      iSetup.get<CSCSurveyRcd>().get(cscSurveys);
      iSetup.get<CSCSurveyErrorExtendedRcd>().get(cscSurveyErrors);
      
      theSurveyIndex  = 0;
      theSurveyValues = &*dtSurveys;
      theSurveyErrors = &*dtSurveyErrors;
      std::vector<Alignable*> barrels = theAlignableMuon->DTBarrel();
      for (std::vector<Alignable*>::const_iterator iter = barrels.begin();  iter != barrels.end();  ++iter) {
	addSurveyInfo_(*iter);
      }
      
      theSurveyIndex  = 0;
      theSurveyValues = &*cscSurveys;
      theSurveyErrors = &*cscSurveyErrors;
      std::vector<Alignable*> endcaps = theAlignableMuon->CSCEndcaps();
      for (std::vector<Alignable*>::const_iterator iter = endcaps.begin();  iter != endcaps.end();  ++iter) {
	addSurveyInfo_(*iter);
      }
    }
  }

}


//////////////////////////////////////////////////
// a templated method - but private, so not accessible from outside
// ==> does not have to be in header file
template<class G, class Rcd, class ErrRcd>
void AlignmentProducer::applyDB(G* geometry, const edm::EventSetup &iSetup,
				const AlignTransform &globalCoordinates) const
{
  // 'G' is the geometry class for that DB should be applied,
  // 'Rcd' is the record class for its Alignments 
  // 'ErrRcd' is the record class for its AlignmentErrorsExtended
  // 'globalCoordinates' are global transformation for this geometry

  const Rcd & record = iSetup.get<Rcd>();
  if (checkDbAlignmentValidity_) {
    const edm::ValidityInterval & validity = record.validityInterval();
    const edm::IOVSyncValue first = validity.first();
    const edm::IOVSyncValue last = validity.last();
    if (first!=edm::IOVSyncValue::beginOfTime() ||
	last!=edm::IOVSyncValue::endOfTime()) {
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
  iSetup.get<ErrRcd>().get(alignmentErrors);

  GeometryAligner aligner;
  aligner.applyAlignments<G>(geometry, &(*alignments), &(*alignmentErrors),
			     globalCoordinates);
}


//////////////////////////////////////////////////
// a templated method - but private, so not accessible from outside
// ==> does not have to be in header file
template<class G, class DeformationRcd>
void AlignmentProducer::applyDB(G* geometry, const edm::EventSetup &iSetup) const
{
  // 'G' is the geometry class for that DB should be applied,
  // 'DeformationRcd' is the record class for its surface deformations 

  const DeformationRcd & record = iSetup.get<DeformationRcd>();
  if (checkDbAlignmentValidity_) {
    const edm::ValidityInterval & validity = record.validityInterval();
    const edm::IOVSyncValue first = validity.first();
    const edm::IOVSyncValue last = validity.last();
    if (first!=edm::IOVSyncValue::beginOfTime() ||
	last!=edm::IOVSyncValue::endOfTime()) {
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

//////////////////////////////////////////////////
void AlignmentProducer::writeForRunRange(cond::Time_t time)
{
  if ( doTracker_ ) { // first tracker
    const AlignTransform *trackerGlobal = 0; // will be 'removed' from constants 
    if (globalPositions_) { // i.e. applied before in applyDB
      trackerGlobal = &align::DetectorGlobalPosition(*globalPositions_,
						     DetId(DetId::Tracker));
    }
	
    Alignments *alignments = theAlignableTracker->alignments();
    AlignmentErrorsExtended *alignmentErrors = theAlignableTracker->alignmentErrors();
    this->writeDB(alignments, "TrackerAlignmentRcd",
		  alignmentErrors, "TrackerAlignmentErrorExtendedRcd", trackerGlobal,
		  time);	
  }
      
  if ( doMuon_ ) { // now muon
    const AlignTransform *muonGlobal = 0; // will be 'removed' from constants 
    if (globalPositions_) { // i.e. applied before in applyDB
      muonGlobal = &align::DetectorGlobalPosition(*globalPositions_,
						  DetId(DetId::Muon));
    }
    // Get alignments+errors, first DT - ownership taken over by writeDB(..), so no delete
    Alignments      *alignments       = theAlignableMuon->dtAlignments();
    AlignmentErrorsExtended *alignmentErrors  = theAlignableMuon->dtAlignmentErrorsExtended();
    this->writeDB(alignments, "DTAlignmentRcd",
		  alignmentErrors, "DTAlignmentErrorExtendedRcd", muonGlobal,
		  time);
    
    // Get alignments+errors, now CSC - ownership taken over by writeDB(..), so no delete
    alignments       = theAlignableMuon->cscAlignments();
    alignmentErrors  = theAlignableMuon->cscAlignmentErrorsExtended();
    this->writeDB(alignments, "CSCAlignmentRcd",
		  alignmentErrors, "CSCAlignmentErrorExtendedRcd", muonGlobal,
		  time);
  }
      
  // Save surface deformations to database
  if (saveDeformationsToDB_ && doTracker_) {
    AlignmentSurfaceDeformations *alignmentSurfaceDeformations = theAlignableTracker->surfaceDeformations();
    this->writeDB(alignmentSurfaceDeformations, "TrackerSurfaceDeformationRcd", time);
  }
}

//////////////////////////////////////////////////
void AlignmentProducer::writeDB(Alignments *alignments,
				const std::string &alignRcd,
				AlignmentErrorsExtended *alignmentErrors,
				const std::string &errRcd,
				const AlignTransform *globalCoordinates,
				cond::Time_t time) const
{
  Alignments * tempAlignments = alignments;
  AlignmentErrorsExtended * tempAlignmentErrorsExtended = alignmentErrors;

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

    edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::writeDB"
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


//////////////////////////////////////////////////
void AlignmentProducer::writeDB(AlignmentSurfaceDeformations *alignmentSurfaceDeformations,
				const std::string &surfaceDeformationRcd,
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

AlignmentProducer::RunRanges
AlignmentProducer::makeNonOverlappingRunRanges(const edm::VParameterSet& RunRangeSelectionVPSet)
{
  static bool oldRunRangeSelectionWarning = false;

  const RunNumber beginValue = cond::timeTypeSpecs[cond::runnumber].beginValue;
  const RunNumber endValue = cond::timeTypeSpecs[cond::runnumber].endValue;
  
  RunRanges uniqueRunRanges;
  if (!RunRangeSelectionVPSet.empty()) {

    std::map<RunNumber,RunNumber> uniqueFirstRunNumbers;
    
    for (std::vector<edm::ParameterSet>::const_iterator ipset = RunRangeSelectionVPSet.begin();
	 ipset != RunRangeSelectionVPSet.end();
	 ++ipset) {
      const std::vector<std::string> RunRangeStrings = (*ipset).getParameter<std::vector<std::string> >("RunRanges");
      for (std::vector<std::string>::const_iterator irange = RunRangeStrings.begin();
	   irange != RunRangeStrings.end();
	   ++irange) {
	
	if ((*irange).find(':')==std::string::npos) {
	  
	  RunNumber first = beginValue;
	  long int temp = strtol((*irange).c_str(), 0, 0);
	  if (temp!=-1) first = temp;
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
	  if (temp!=-1) first = temp;
	  uniqueFirstRunNumbers[first] = first;
	}
      }
    }

    for (std::map<RunNumber,RunNumber>::iterator iFirst = uniqueFirstRunNumbers.begin();
	 iFirst!=uniqueFirstRunNumbers.end();
	 ++iFirst) {
      uniqueRunRanges.push_back(std::pair<RunNumber,RunNumber>((*iFirst).first, endValue));
    }
    for (unsigned int i = 0;i<uniqueRunRanges.size()-1;++i) {
      uniqueRunRanges[i].second = uniqueRunRanges[i+1].first - 1;
    }
    
  } else {
        
    uniqueRunRanges.push_back(std::pair<RunNumber,RunNumber>(beginValue, endValue));
    
  }
  
  return uniqueRunRanges;
}

DEFINE_FWK_LOOPER( AlignmentProducer );
