/// \file AlignmentProducer.cc
///
///  \author    : Frederic Ronga
///  Revision   : $Revision: 1.27 $
///  last update: $Date: 2007/03/13 01:50:03 $
///  by         : $Author: cklae $

#include "Alignment/CommonAlignmentProducer/interface/AlignmentProducer.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

// System include files
#include <memory>
#include <sstream>

// Framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "Geometry/DTGeometryBuilder/src/DTGeometryBuilderFromDDD.h"
#include "Geometry/CSCGeometryBuilder/src/CSCGeometryBuilderFromDDD.h"
#include "Geometry/TrackingGeometryAligner/interface/GeometryAligner.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/TrackerAlignmentErrorRcd.h"
#include "CondFormats/DataRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/DataRecord/interface/CSCAlignmentRcd.h"
#include "CondFormats/DataRecord/interface/CSCAlignmentErrorRcd.h"

// Tracking 	 
#include "TrackingTools/PatternTools/interface/Trajectory.h" 

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "Alignment/TrackerAlignment/interface/TrackerScenarioBuilder.h"
#include "Alignment/MuonAlignment/interface/MuonScenarioBuilder.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"

//_____________________________________________________________________________
AlignmentProducer::AlignmentProducer(const edm::ParameterSet& iConfig) :
  theAlignableTracker(0),theAlignableMuon(0),
  theMaxLoops( iConfig.getUntrackedParameter<unsigned int>("maxLoops",0) ),
  stNFixAlignables_(iConfig.getParameter<int>("nFixAlignables") ),
  stRandomShift_(iConfig.getParameter<double>("randomShift")),
  stRandomRotation_(iConfig.getParameter<double>("randomRotation")),
  applyDbAlignment_( iConfig.getUntrackedParameter<bool>("applyDbAlignment",false) ),
  doMisalignmentScenario_(iConfig.getParameter<bool>("doMisalignmentScenario")),
  saveToDB_(iConfig.getParameter<bool>("saveToDB")),
  doTracker_( iConfig.getUntrackedParameter<bool>("doTracker") ),
  doMuon_( iConfig.getUntrackedParameter<bool>("doMuon") )
{

  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::AlignmentProducer";

  theParameterSet=iConfig;

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
  std::string algoName = algoConfig.getParameter<std::string>("algoName");
  theAlignmentAlgo = AlignmentAlgorithmPluginFactory::getAlgorithm( algoName, algoConfig );

  // Check if found
  if ( !theAlignmentAlgo )
	throw cms::Exception("BadConfig") << "Couldn't find algorithm called " << algoName;
}


//_____________________________________________________________________________
// Delete new objects
AlignmentProducer::~AlignmentProducer()
{

  delete theAlignmentParameterStore;
  delete theAlignmentParameterBuilder;

  if (theAlignableTracker) delete theAlignableTracker;
  if (theAlignableMuon)    delete theAlignableMuon;

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

  nevent_ = 0;
  GeometryAligner aligner;

  // Create the geometries from the ideal geometries (first time only)
  this->createGeometries_( iSetup );
  
  // Retrieve and apply alignments, if requested (requires DB setup)
  if ( applyDbAlignment_ ) {
    if ( doTracker_ ) {
      edm::ESHandle<Alignments> alignments;
      iSetup.get<TrackerAlignmentRcd>().get( alignments );
      edm::ESHandle<AlignmentErrors> alignmentErrors;
      iSetup.get<TrackerAlignmentErrorRcd>().get( alignmentErrors );
      aligner.applyAlignments<TrackerGeometry>( &(*theTracker), &(*alignments), &(*alignmentErrors) );
    }
    if ( doMuon_ ) {
      edm::ESHandle<Alignments> dtAlignments;
      iSetup.get<DTAlignmentRcd>().get( dtAlignments );
      edm::ESHandle<AlignmentErrors> dtAlignmentErrors;
      iSetup.get<DTAlignmentErrorRcd>().get( dtAlignmentErrors );
      aligner.applyAlignments<DTGeometry>( &(*theMuonDT), &(*dtAlignments), &(*dtAlignmentErrors) );

      edm::ESHandle<Alignments> cscAlignments;
      iSetup.get<CSCAlignmentRcd>().get( cscAlignments );
      edm::ESHandle<AlignmentErrors> cscAlignmentErrors;
      iSetup.get<CSCAlignmentErrorRcd>().get( cscAlignmentErrors );
      aligner.applyAlignments<CSCGeometry>( &(*theMuonCSC), &(*cscAlignments), &(*cscAlignmentErrors) );
    }
  }

  // Create alignable tracker and muon 
  if (doTracker_) theAlignableTracker = new AlignableTracker( &(*theGeometricDet), &(*theTracker) );
  if (doMuon_) theAlignableMuon = new AlignableMuon( &(*theMuonDT), &(*theMuonCSC) );

  // Create alignment parameter builder
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob" 
                            << "Creating AlignmentParameterBuilder";
  edm::ParameterSet aliParamBuildCfg = 
    theParameterSet.getParameter<edm::ParameterSet>("ParameterBuilder");
  theAlignmentParameterBuilder = new AlignmentParameterBuilder( theAlignableTracker,
                                                                theAlignableMuon,
                                                                aliParamBuildCfg );
  // Fix alignables if requested
  if (stNFixAlignables_>0) theAlignmentParameterBuilder->fixAlignables(stNFixAlignables_);

  // Get list of alignables
  Alignables theAlignables = theAlignmentParameterBuilder->alignables();
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

  // Initialize alignment algorithm
  theAlignmentAlgo->initialize( iSetup, theAlignableTracker,
                                theAlignableMuon, theAlignmentParameterStore );
}

//_____________________________________________________________________________
// Terminate algorithm
void AlignmentProducer::endOfJob()
{


  // Save alignments to database
  if (saveToDB_) {
    edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::endOfJob" 
                              << "Writing Alignments to DB...";
    // Call service
    edm::Service<cond::service::PoolDBOutputService> poolDbService;
    if( !poolDbService.isAvailable() ) // Die if not available
      throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
    
    if ( doTracker_ ) {
       // Get alignments+errors
       Alignments* alignments = theAlignableTracker->alignments();
       AlignmentErrors* alignmentErrors = theAlignableTracker->alignmentErrors();

       // Store
       std::string alignRecordName( "TrackerAlignmentRcd" );
       std::string errorRecordName( "TrackerAlignmentErrorRcd" );

       if ( poolDbService->isNewTagRequest(alignRecordName) )
	  poolDbService->createNewIOV<Alignments>( alignments, poolDbService->endOfTime(), 
						   alignRecordName );
       else
	  poolDbService->appendSinceTime<Alignments>( alignments, poolDbService->currentTime(), 
						      alignRecordName );
       if ( poolDbService->isNewTagRequest(errorRecordName) )
	  poolDbService->createNewIOV<AlignmentErrors>( alignmentErrors, poolDbService->endOfTime(), 
							errorRecordName );
       else
	  poolDbService->appendSinceTime<AlignmentErrors>( alignmentErrors, poolDbService->currentTime(), 
							   errorRecordName );
    }
 
    if ( doMuon_ ) {
       // Get alignments+errors
       Alignments*      dtAlignments       = theAlignableMuon->dtAlignments();
       AlignmentErrors* dtAlignmentErrors  = theAlignableMuon->dtAlignmentErrors();
       Alignments*      cscAlignments      = theAlignableMuon->cscAlignments();
       AlignmentErrors* cscAlignmentErrors = theAlignableMuon->cscAlignmentErrors();

       std::string dtAlignRecordName( "DTAlignments" );
       std::string dtErrorRecordName( "DTAlignmentErrors" );
       std::string cscAlignRecordName( "CSCAlignments" );
       std::string cscErrorRecordName( "CSCAlignmentErrors" );

       if (poolDbService->isNewTagRequest(dtAlignRecordName)) {
	  poolDbService->createNewIOV<Alignments>( &(*dtAlignments), poolDbService->endOfTime(), dtAlignRecordName);
       }
       else {
	  poolDbService->appendSinceTime<Alignments>( &(*dtAlignments), poolDbService->currentTime(), dtAlignRecordName);
       }
       if (poolDbService->isNewTagRequest(dtErrorRecordName)) {
	  poolDbService->createNewIOV<AlignmentErrors>( &(*dtAlignmentErrors), poolDbService->endOfTime(), dtErrorRecordName);
       }
       else {
	  poolDbService->appendSinceTime<AlignmentErrors>( &(*dtAlignmentErrors), poolDbService->currentTime(), dtErrorRecordName);
       }
       if (poolDbService->isNewTagRequest(cscAlignRecordName)) {
	  poolDbService->createNewIOV<Alignments>( &(*cscAlignments), poolDbService->endOfTime(), cscAlignRecordName);
       }
       else {
	  poolDbService->appendSinceTime<Alignments>( &(*cscAlignments), poolDbService->currentTime(), cscAlignRecordName);
       }
       if (poolDbService->isNewTagRequest(cscErrorRecordName)) {
	  poolDbService->createNewIOV<AlignmentErrors>( &(*cscAlignmentErrors), poolDbService->endOfTime(), cscErrorRecordName);
       }
       else {
	  poolDbService->appendSinceTime<AlignmentErrors>( &(*cscAlignmentErrors), poolDbService->currentTime(), cscErrorRecordName);
       }
    }
  }
}

//_____________________________________________________________________________
// Called at beginning of loop
void AlignmentProducer::startingNewLoop(unsigned int iLoop )
{
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::startingNewLoop" 
                            << "Starting loop number " << iLoop;

  theAlignmentAlgo->startNewLoop();

  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::startingNewLoop" 
                            << "Now physically apply alignments to  geometry...";


  // Propagate changes to reconstruction geometry (from initialisation or iteration)
  GeometryAligner aligner;
  if ( doTracker_ ) {
    std::auto_ptr<Alignments> alignments(theAlignableTracker->alignments());
    std::auto_ptr<AlignmentErrors> alignmentErrors(theAlignableTracker->alignmentErrors());
    aligner.applyAlignments<TrackerGeometry>( &(*theTracker),&(*alignments),&(*alignmentErrors));
  }
  if ( doMuon_ ) {
    std::auto_ptr<Alignments>      dtAlignments(       theAlignableMuon->dtAlignments());
    std::auto_ptr<AlignmentErrors> dtAlignmentErrors(  theAlignableMuon->dtAlignmentErrors());
    std::auto_ptr<Alignments>      cscAlignments(      theAlignableMuon->cscAlignments());
    std::auto_ptr<AlignmentErrors> cscAlignmentErrors( theAlignableMuon->cscAlignmentErrors());

    aligner.applyAlignments<DTGeometry>( &(*theMuonDT), &(*dtAlignments), &(*dtAlignmentErrors) );
    aligner.applyAlignments<CSCGeometry>( &(*theMuonCSC), &(*cscAlignments), &(*cscAlignmentErrors) );
  }
}


//_____________________________________________________________________________
// Called at end of loop
edm::EDLooper::Status 
AlignmentProducer::endOfLoop(const edm::EventSetup& iSetup, unsigned int iLoop)
{
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::endOfLoop" 
                            << "Ending loop " << iLoop;

  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::endOfLoop" 
                            << "Terminating algorithm.";
  theAlignmentAlgo->terminate();

  if ( iLoop == theMaxLoops-1 || iLoop >= theMaxLoops ) return kStop;
  else return kContinue;
}

//_____________________________________________________________________________
// Called at each event
edm::EDLooper::Status 
AlignmentProducer::duringLoop( const edm::Event& event, 
  const edm::EventSetup& setup )
{
  nevent_++;

  for ( int i=10; i<10000000; i*=10 )
    if ( nevent_<10*i && (nevent_%i)==0 )
      edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::duringLoop" 
                                << "Events processed: " << nevent_;

  // Retrieve trajectories and tracks from the event
  edm::InputTag tjTag = theParameterSet.getParameter<edm::InputTag>("tjTkAssociationMapTag");
  edm::Handle<TrajTrackAssociationCollection> m_TrajTracksMap;
  event.getByLabel( tjTag, m_TrajTracksMap );

  // Form pairs of trajectories and tracks
  ConstTrajTrackPairCollection trajTracks;
  for ( TrajTrackAssociationCollection::const_iterator iPair = m_TrajTracksMap->begin();
        iPair != m_TrajTracksMap->end(); iPair++ )
    trajTracks.push_back( ConstTrajTrackPair( &(*(*iPair).key), &(*(*iPair).val) ) );

  // Run the alignment algorithm
  theAlignmentAlgo->run(  setup, trajTracks );

  return kContinue;
}

// ----------------------------------------------------------------------------

void AlignmentProducer::simpleMisalignment_(const Alignables &alivec, const std::string &selection, 
                                            float shift, float rot, bool local)
{

  std::ostringstream output; // collecting output

  if (shift>0 || rot >0) {
    output << "Adding random flat shift of max size " << shift
           << " and adding random flat rotation of max size " << rot <<" to ";

    std::vector<bool> commSel(0);
    if (selection != "-1") {
      AlignmentParameterSelector aSelector(0,0); // no alignable needed here...
      const std::vector<char> cSel(aSelector.convertParamSel(selection));
      for (std::vector<char>::const_iterator cIter = cSel.begin(); cIter != cSel.end(); ++cIter) {
        commSel.push_back(*cIter == '0' ? false : true);
      }
      output << "parameters defined by (" << selection 
             << "), representing (x,y,z,alpha,beta,gamma).";
    } else {
      output << "the active parameters of each alignable.";
    }

    for (std::vector<Alignable*>::const_iterator it = alivec.begin(); it != alivec.end(); ++it) {
      Alignable* ali=(*it);
      std::vector<bool> mysel(commSel.empty() ? ali->alignmentParameters()->selector() : commSel);

      if (std::abs(shift)>0.00001) {

        double s0 = mysel[RigidBodyAlignmentParameters::dx] ?
	  shift*double(random()%1000-500)/500. : 0.;

        double s1 = mysel[RigidBodyAlignmentParameters::dy] ?
          shift*double(random()%1000-500)/500. : 0.;

        double s2 = mysel[RigidBodyAlignmentParameters::dz] ?
          shift*double(random()%1000-500)/500. : 0.;
        
        if (local)
          ali->move( ali->surface().toGlobal(align::LocalVector(s0,s1,s2)) );
	else
          ali->move( align::GlobalVector(s0,s1,s2) );

      //AlignmentPositionError ape(dx,dy,dz);
      //ali->addAlignmentPositionError(ape);
      }

      if (std::abs(rot)>0.00001) {
	align::EulerAngles r(3);

        if (mysel[RigidBodyAlignmentParameters::dalpha]) {
          r(1)=rot*double(random()%1000-500)/500.;
        }
        if (mysel[RigidBodyAlignmentParameters::dbeta]) {
          r(2)=rot*double(random()%1000-500)/500.;
        }
        if (mysel[RigidBodyAlignmentParameters::dgamma]) {
          r(3)=rot*double(random()%1000-500)/500.;
        }

        align::RotationType mrot = align::toMatrix(r);
        if (local) ali->rotateInLocalFrame(mrot);
        else ali->rotateInGlobalFrame(mrot);
        
      //ali->addAlignmentPositionErrorFromRotation(mrot);

      }
    } // end loop on alignables
  } else {
    output << "No simple misalignment added!";
  }
  edm::LogWarning("Alignment")  << "@SUB=AlignmentProducer::simpleMisalignment_" << output.str();

}


//__________________________________________________________________________________________________
void AlignmentProducer::createGeometries_( const edm::EventSetup& iSetup )
{
   edm::ESHandle<DDCompactView> cpv;
   iSetup.get<IdealGeometryRecord>().get( cpv );

   if (doTracker_) {
      iSetup.get<IdealGeometryRecord>().get( theGeometricDet );
      TrackerGeomBuilderFromGeometricDet trackerBuilder;
      theTracker = boost::shared_ptr<TrackerGeometry>( trackerBuilder.build(&(*cpv),
									    &(*theGeometricDet)) );
   }

   if (doMuon_) {
      edm::ESHandle<MuonDDDConstants> mdc;
      iSetup.get<MuonNumberingRecord>().get(mdc);
      DTGeometryBuilderFromDDD DTGeometryBuilder;
      CSCGeometryBuilderFromDDD CSCGeometryBuilder;
      theMuonDT = boost::shared_ptr<DTGeometry>(DTGeometryBuilder.build(&(*cpv), *mdc));
      theMuonCSC = boost::shared_ptr<CSCGeometry>(CSCGeometryBuilder.build(&(*cpv), *mdc));
   }
}
