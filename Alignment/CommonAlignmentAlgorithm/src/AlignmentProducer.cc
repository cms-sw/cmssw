
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentProducer.h"

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "Alignment/CSA06AlignmentAlgorithm/interface/CSA06AlignmentAlgorithm.h"

//_____________________________________________________________________________

AlignmentProducer::AlignmentProducer(const edm::ParameterSet& iConfig) : 
  theRefitterAlgo(iConfig)
{
  setConf( iConfig );
  setSrc( iConfig.getParameter<std::string>( "src" ) );

  // Register your products
  produces<TrackingRecHitCollection>();
  produces<reco::TrackCollection>();
  produces<reco::TrackExtraCollection>();

  std::cout <<"[AlignmentProducer::AlignmentProducer] called\n";

  // create alignable tracker
  //theAlignableTracker = new AlignableTracker();

  // create alignment algorithm
  edm::ParameterSet csa06Config = iConfig.getParameter<edm::ParameterSet>(
    "CSA06AlignmentAlgorithm" );
  theAlignmentAlgo = new CSA06AlignmentAlgorithm(csa06Config,theAlignableTracker);

}

//_____________________________________________________________________________
// Initialize algorithm

void AlignmentProducer::beginJob(EventSetup const& setup)
{
  std::cout <<"[AlignmentProducer::beginJob] called\n";
  theAlignmentAlgo->initialize( setup );
}

//_____________________________________________________________________________
// Terminate algorithm

void AlignmentProducer::endJob()
{
  std::cout <<"[AlignmentProducer::endJob] called\n";
  theAlignmentAlgo->terminate();
}

//_____________________________________________________________________________
// Called at each event

void AlignmentProducer::produce(edm::Event& event, 
  const edm::EventSetup& setup)
{

  edm::LogInfo("TrackProducer") << "Analyzing event number: " << event.id();

  //Create empty output collections
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
  getFromEvt(event, m_TrackCollection);
    
  // Run the refitter algorithm  
  AlgoProductCollection m_algoResults;
  theRefitterAlgo.runWithTrack( m_Geometry.product(),m_MagneticField.product(),
    *m_TrackCollection, m_TrajectoryFitter.product(), m_Propagator.product(), 
    m_RecHitBuilder.product(), m_algoResults );

  // Strip out the tracks to keep trajectories only
  //std::vector<Trajectory*> trajectories = this->getTrajectories( m_algoResults );

  // Run the alignment algorithm
  theAlignmentAlgo->run(  m_algoResults );
  //theAlignmentAlgo->run( trajectories );

  // Put everything in the event => WHAT?
  putInEvt( event, outputRHColl, outputTColl, outputTEColl, m_algoResults );
  LogDebug("TrackProducer") << "end";

}

//_____________________________________________________________________________

// Keep only trajectories from pairs of trajectories/initial tracks
// From TrackProducerAlgorithm.h: 
// typedef std::pair<Trajectory*, reco::Track*> AlgoProduct; 
// typedef std::vector< AlgoProduct >  AlgoProductCollection;

std::vector<Trajectory*> 
AlignmentProducer::getTrajectories( AlgoProductCollection algoResults )
{
  std::vector<Trajectory*> result;
  for ( AlgoProductCollection::iterator iPair = algoResults.begin();
		iPair != algoResults.end(); iPair++ )
	result.push_back( (*iPair).first );
  return result;
}

