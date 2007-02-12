/// \file AlignmentProducer.cc
///
///  \author    : Frederic Ronga
///  Revision   : $Revision: 1.22 $
///  last update: $Date: 2007/01/26 17:19:47 $
///  by         : $Author: flucke $

#include "Alignment/CommonAlignmentProducer/interface/AlignmentProducer.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h" 

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
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

// Alignment
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrors.h"
#include "Alignment/TrackerAlignment/interface/TrackerScenarioBuilder.h"
#include "Alignment/MuonAlignment/interface/MuonScenarioBuilder.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentTransformations.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmPluginFactory.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"

using namespace std;

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
  setWhatProduced(this, &AlignmentProducer::produceTracker);
  setWhatProduced(this, &AlignmentProducer::produceDT);
  setWhatProduced(this, &AlignmentProducer::produceCSC);

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
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::beginOfJob" 
                            << "Now physically apply alignments to  geometry...";

  // Actually execute all misalignments
  if ( doTracker_ ) {
    std::auto_ptr<Alignments> alignments(theAlignableTracker->alignments());
    std::auto_ptr<AlignmentErrors> alignmentErrors(theAlignableTracker->alignmentErrors());
    aligner.applyAlignments<TrackerGeometry>( &(*theTracker),&(*alignments),&(*alignmentErrors));
  }
  if ( doMuon_ ) {
    std::auto_ptr<Alignments> dtAlignments(theAlignableTracker->alignments());
    std::auto_ptr<AlignmentErrors> dtAlignmentErrors(theAlignableTracker->alignmentErrors());
    aligner.applyAlignments<DTGeometry>( &(*theMuonDT), &(*dtAlignments), &(*dtAlignmentErrors) );
    std::auto_ptr<Alignments> cscAlignments(theAlignableTracker->alignments());
    std::auto_ptr<AlignmentErrors> cscAlignmentErrors(theAlignableTracker->alignmentErrors());
    aligner.applyAlignments<CSCGeometry>( &(*theMuonCSC), &(*cscAlignments), &(*cscAlignmentErrors) );
  }
}

//_____________________________________________________________________________
// Terminate algorithm
void AlignmentProducer::endOfJob()
{

  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::endOfJob" 
                            << "Terminating algorithm.";
  theAlignmentAlgo->terminate();

  // Save alignments to database
  if (saveToDB_) {
    edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::endOfJob" 
                              << "Writing Alignments to DB...";
    // Call service
    edm::Service<cond::service::PoolDBOutputService> poolDbService;
    if( !poolDbService.isAvailable() ) // Die if not available
      throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
    
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


}

//_____________________________________________________________________________
// Called at beginning of loop
void AlignmentProducer::startingNewLoop(unsigned int iLoop )
{
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::startingNewLoop" 
                            << "Starting loop number " << iLoop;
}


//_____________________________________________________________________________
// Called at end of loop
edm::EDLooper::Status 
AlignmentProducer::endOfLoop(const edm::EventSetup& iSetup, unsigned int iLoop)
{
  edm::LogInfo("Alignment") << "@SUB=AlignmentProducer::endOfLoop" 
                            << "Ending loop " << iLoop;

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
  edm::InputTag tkTag = theParameterSet.getParameter<edm::InputTag>("tkTag");
  edm::Handle<reco::TrackCollection> m_TrackCollection;
  event.getByLabel( tkTag, m_TrackCollection );
  edm::InputTag tjTag = theParameterSet.getParameter<edm::InputTag>("tjTag");
  edm::Handle<std::vector<Trajectory> > m_TrajectoryCollection;
  event.getByLabel( tjTag, m_TrajectoryCollection );

  // Form pairs of trajectories and tracks
  ConstTrajTrackPairCollection m_algoResults;
  reco::TrackCollection::const_iterator   iTrack = m_TrackCollection->begin();
  std::vector<Trajectory>::const_iterator iTraj  = m_TrajectoryCollection->begin();
  for ( ; iTrack != m_TrackCollection->end(); ++iTrack, ++iTraj )
    {
      ConstTrajTrackPair aPair(  &(*iTraj), &(*iTrack)  );
      if ( !this->trajTrackMatch_( aPair ) )
        throw cms::Exception("TrajTrackMismatch") << "Couldn't pair trajectory and track";
      m_algoResults.push_back( aPair );
    }

  // Run the alignment algorithm
  theAlignmentAlgo->run(  setup, m_algoResults );


//   // Retrieve trajectories and tracks from the event
//   edm::InputTag tkTag = theParameterSet.getParameter<edm::InputTag>("tkTag");
//   edm::Handle<TrajTrackAssociationCollection> m_TrajTracksMap;
//   event.getByLabel( tkTag, m_TrajTracksMap );

//   // Form pairs of trajectories and tracks
//   ConstTrajTrackPairCollection trajTracks;
//   for ( TrajTrackAssociationCollection::const_iterator iPair = m_TrajTracksMap->begin();
//         iPair != m_TrajTracksMap->end(); iPair++ )
//     trajTracks.push_back( ConstTrajTrackPair( &(*(*iPair).key), &(*(*iPair).val) ) );

//   // Run the alignment algorithm
//   theAlignmentAlgo->run(  setup, trajTracks );

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

    for (vector<Alignable*>::const_iterator it = alivec.begin(); it != alivec.end(); ++it) {
      Alignable* ali=(*it);
      vector<bool> mysel(commSel.empty() ? ali->alignmentParameters()->selector() : commSel);

      if (abs(shift)>0.00001) {
        AlgebraicVector s(3);
        s[0]=0; s[1]=0; s[2]=0;  
        if (mysel[RigidBodyAlignmentParameters::dx]) {
          s[0]=shift*double(random()%1000-500)/500.;
        }
        if (mysel[RigidBodyAlignmentParameters::dy]) {
          s[1]=shift*double(random()%1000-500)/500.;
        }
        if (mysel[RigidBodyAlignmentParameters::dz]) {
          s[2]=shift*double(random()%1000-500)/500.;
        }
        
        GlobalVector globalshift;
        if (local) {
          globalshift = ali->surface().toGlobal(Local3DVector(s[0],s[1],s[2]));
        } else {
          globalshift = Global3DVector(s[0],s[1],s[2]);
        }
        ali->move(globalshift);

      //AlignmentPositionError ape(dx,dy,dz);
      //ali->addAlignmentPositionError(ape);
      }

      if (abs(rot)>0.00001) {
        AlgebraicVector r(3);
        r[0]=0; r[1]=0; r[2]=0;
        if (mysel[RigidBodyAlignmentParameters::dalpha]) {
          r[0]=rot*double(random()%1000-500)/500.;
        }
        if (mysel[RigidBodyAlignmentParameters::dbeta]) {
          r[1]=rot*double(random()%1000-500)/500.;
        }
        if (mysel[RigidBodyAlignmentParameters::dgamma]) {
          r[2]=rot*double(random()%1000-500)/500.;
        }
        AlignmentTransformations TkAT;
        Surface::RotationType mrot = TkAT.rotationType(TkAT.rotMatrix3(r));
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
  iSetup.get<IdealGeometryRecord>().get( theGeometricDet );
  TrackerGeomBuilderFromGeometricDet trackerBuilder;
  theTracker = boost::shared_ptr<TrackerGeometry>( trackerBuilder.build(&(*cpv),
                                                                        &(*theGeometricDet)) );

  edm::ESHandle<MuonDDDConstants> mdc;
  iSetup.get<MuonNumberingRecord>().get(mdc);
  DTGeometryBuilderFromDDD DTGeometryBuilder;
  CSCGeometryBuilderFromDDD CSCGeometryBuilder;
  theMuonDT = boost::shared_ptr<DTGeometry>(DTGeometryBuilder.build(&(*cpv), *mdc));
  theMuonCSC = boost::shared_ptr<CSCGeometry>(CSCGeometryBuilder.build(&(*cpv), *mdc));

}


//__________________________________________________________________________________________________
const bool AlignmentProducer::trajTrackMatch_( const ConstTrajTrackPair& pair ) const
{

  // Compare a trajectory and a track
  // Currently based on rec.hits. comparison


  // 1. - should have same number of hits
  if ( pair.first->measurements().size() != pair.second->recHitsSize() ) return false;

  // 2. - compare hits
  Trajectory::ConstRecHitContainer recHits( pair.first->recHits() );
  trackingRecHit_iterator iTkHit = pair.second->recHitsBegin();

  for ( Trajectory::ConstRecHitContainer::const_iterator iTjHit = recHits.begin();
        iTjHit != recHits.end(); ++iTjHit, ++iTkHit )
    {

      if ( (*iTjHit)->isValid() && (*iTkHit)->isValid() ) // Skip invalid hits
        {

          // Module Id
          if ( (*iTjHit)->geographicalId() != (*iTkHit)->geographicalId() )
            return false;

          // Local position
          if ( fabs((*iTjHit)->localPosition().x() - (*iTkHit)->localPosition().x()) > 1.e-12
               || fabs((*iTjHit)->localPosition().y() - (*iTkHit)->localPosition().y()) > 1.e-12
               || fabs((*iTjHit)->localPosition().z() - (*iTkHit)->localPosition().z()) > 1.e-12
               )
            return false;
        }
    }

  return true;

}
