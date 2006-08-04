
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentProducer.h"

// System include files
#include <memory>

// Framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeomBuilderFromGeometricDet.h"

#include "Alignment/CSA06AlignmentAlgorithm/interface/CSA06AlignmentAlgorithm.h"

//_____________________________________________________________________________

AlignmentProducer::AlignmentProducer(const edm::ParameterSet& iConfig) :
  theRefitterAlgo( iConfig ),
  theMaxLoops( iConfig.getUntrackedParameter<unsigned int>("maxLoops",0) ),
  theSrc( iConfig.getParameter<std::string>( "src" ) )
{

  edm::LogInfo("Constructor") << "Constructing producer";

  // Tell the framework what data is being produced
  setWhatProduced(this);

  setConf( iConfig );
  setSrc( iConfig.getParameter<std::string>( "src" ) );

  // create alignment algorithm
  edm::ParameterSet csa06Config 
	= iConfig.getParameter<edm::ParameterSet>( "CSA06AlignmentAlgorithm" );

  theAlignmentAlgo = new CSA06AlignmentAlgorithm(csa06Config);

}


//__________________________________________________________________________________________________
// Close files, etc.
AlignmentProducer::~AlignmentProducer()
{

}


//__________________________________________________________________________________________________
// Produce tracker geometry
AlignmentProducer::ReturnType 
AlignmentProducer::produce( const TrackerDigiGeometryRecord& iRecord )
{

  edm::LogInfo("Produce") << "At producer method";

  return theTracker;
  
}


//_____________________________________________________________________________
// Initialize algorithm
void AlignmentProducer::beginOfJob( const edm::EventSetup& iSetup )
{

  edm::LogInfo("BeginJob") << "At begin job";

  // Create the tracker geometry from ideal geometry (first time only)
  edm::ESHandle<DDCompactView> cpv;
  edm::ESHandle<GeometricDet> gD;
  iSetup.get<IdealGeometryRecord>().get( cpv );
  iSetup.get<IdealGeometryRecord>().get( gD );
  TrackerGeomBuilderFromGeometricDet trackerBuilder;
  theTracker  = boost::shared_ptr<TrackerGeometry>( trackerBuilder.build(&(*cpv),&(*gD)) );
  
  // create alignable tracker
  theAlignableTracker = new AlignableTracker( &(*gD), &(*theTracker) );

  theAlignmentAlgo->initialize( iSetup, theAlignableTracker );

}

//_____________________________________________________________________________
// Terminate algorithm
void AlignmentProducer::endOfJob()
{

  edm::LogInfo("EndJob") << "At end of job: terminating algorithm";
  theAlignmentAlgo->terminate();

}


//__________________________________________________________________________________________________
// Called at beginning of loop
void AlignmentProducer::startingNewLoop(unsigned int iLoop )
{

  edm::LogInfo("NewLoop") << "Starting loop number " << iLoop;

}


//__________________________________________________________________________________________________
// Called at end of loop
edm::EDLooper::Status AlignmentProducer::endOfLoop( const edm::EventSetup& iSetup, 
						    unsigned int iLoop )
{
  
  edm::LogInfo("EndLoop") << "Ending loop " << iLoop;

  if ( iLoop == theMaxLoops-1 || iLoop >= theMaxLoops ) return kStop;
  else return kContinue;

}

//_____________________________________________________________________________
// Called at each event
edm::EDLooper::Status 
AlignmentProducer::duringLoop( const edm::Event& event, const edm::EventSetup& setup )
{

  edm::LogInfo("InLoop") << "Analyzing event";

  std::auto_ptr<TrackingRecHitCollection> outputRHColl(new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);

  // Retrieve what we need from the EventSetup
  edm::ESHandle<TrackerGeometry>  m_Geometry;
  edm::ESHandle<MagneticField>    m_MagneticField;
  edm::ESHandle<TrajectoryFitter> m_TrajectoryFitter;
  edm::ESHandle<Propagator>       m_Propagator;
  edm::ESHandle<TransientTrackingRecHitBuilder> m_RecHitBuilder;
  getFromES( setup, m_Geometry, m_MagneticField, m_TrajectoryFitter, 
             m_Propagator, m_RecHitBuilder );

  // Retrieve track collection from the event
  edm::Handle<reco::TrackCollection> m_TrackCollection;
  event.getByLabel( theSrc, m_TrackCollection );
  //getFromEvt( event, m_TrackCollection );
    
  // Run the refitter algorithm  
  AlgoProductCollection m_algoResults;
  theRefitterAlgo.runWithTrack( m_Geometry.product(),m_MagneticField.product(),
    *m_TrackCollection, m_TrajectoryFitter.product(), m_Propagator.product(), 
    m_RecHitBuilder.product(), m_algoResults );

  // Run the alignment algorithm
  theAlignmentAlgo->run(  m_algoResults );


  return kContinue;

}

