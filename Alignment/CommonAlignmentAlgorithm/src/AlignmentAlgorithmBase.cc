#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"


//__________________________________________________________________________________________________
AlignmentAlgorithmBase::AlignmentAlgorithmBase( const edm::ParameterSet& cfg ) :
  theRefitterAlgo( cfg ),
  theSrc( cfg.getParameter<std::string>( "src" ) ),
  debug( cfg.getParameter<bool>("debug") )
{

  TrackProducerBase::setConf( cfg );
  TrackProducerBase::setSrc( cfg.getParameter<std::string>( "src" ) );

}

//__________________________________________________________________________________________________
AlgoProductCollection 
AlignmentAlgorithmBase::refitTracks( const edm::Event& event, const edm::EventSetup& setup )
{

  // This piece of code is copied from the TrackRefitter available in
  // RecoTracker/TrackProducer. Refitting is run locally to:
  // - get access to the TSOS (not stored in the event yet)
  // - allow the KF algo. to run its own updator.
  

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

  // Dump original tracks
  if ( debug ) {
	printf("Original tracks:\n");
	for( reco::TrackCollection::const_iterator itrack = m_TrackCollection->begin(); 
		 itrack != m_TrackCollection->end(); ++ itrack ) {
	  reco::Track track=*itrack;
	  printf( "Org track pt,eta,phi,hits,chi2: %12.5f %12.5f %12.5f %5d %12.5f\n",
			  track.pt(), track.eta(), track.phi(), 
			  track.recHitsSize(), track.normalizedChi2() );
	}
  }

  AlgoProductCollection m_algoResults;
  theRefitterAlgo.runWithTrack( m_Geometry.product(),m_MagneticField.product(),
								*m_TrackCollection, 
								m_TrajectoryFitter.product(), m_Propagator.product(), 
								m_RecHitBuilder.product(), m_algoResults );


  // Dump refitted tracks
  if ( debug ) 
	{
	  printf("Refitted tracks:\n");
	  for( AlgoProductCollection::const_iterator it=m_algoResults.begin();
		   it!=m_algoResults.end();it++) {
		Trajectory* traj = (*it).first;
		reco::Track* track = (*it).second;
		printf("Fit track pt,eta,phi,hits,chi2: %12.5f %12.5f %12.5f %5d %12.5f\n",
		       track->pt(), track->eta(), track->phi(), 
			   traj->measurements().size(), track->normalizedChi2() );
	  }
	}

  return m_algoResults;


}
