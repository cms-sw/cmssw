#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"
#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectory.h" 

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"

/// A factory that produces instances of class ReferenceTrajectory from a given TrajTrackPairCollection.

class ReferenceTrajectoryFactory : public TrajectoryFactoryBase
{
public:
  ReferenceTrajectoryFactory(const edm::ParameterSet &config);
  virtual ~ReferenceTrajectoryFactory();

  /// Produce the reference trajectories.
  virtual const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
							   const ConstTrajTrackPairCollection &tracks,
							   const reco::BeamSpot &beamSpot) const;

  virtual const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
							   const ConstTrajTrackPairCollection &tracks,
							   const ExternalPredictionCollection &external,
							   const reco::BeamSpot &beamSpot) const;

  virtual ReferenceTrajectoryFactory* clone() const { return new ReferenceTrajectoryFactory(*this); }

protected:

  double theMass;
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

ReferenceTrajectoryFactory::ReferenceTrajectoryFactory( const edm::ParameterSet & config ) :
  TrajectoryFactoryBase( config )
{
  theMass = config.getParameter< double >( "ParticleMass" );
}

 
ReferenceTrajectoryFactory::~ReferenceTrajectoryFactory( void ) {}


const ReferenceTrajectoryFactory::ReferenceTrajectoryCollection
ReferenceTrajectoryFactory::trajectories(const edm::EventSetup &setup,
					 const ConstTrajTrackPairCollection &tracks,
					 const reco::BeamSpot &beamSpot) const
{
  ReferenceTrajectoryCollection trajectories;

  edm::ESHandle< MagneticField > magneticField;
  setup.get< IdealMagneticFieldRecord >().get( magneticField );
  
  ConstTrajTrackPairCollection::const_iterator itTracks = tracks.begin();
  
  while ( itTracks != tracks.end() )
  { 
    TrajectoryInput input = this->innermostStateAndRecHits( *itTracks );
    
    // Check input: If all hits were rejected, the TSOS is initialized as invalid.
    if ( input.first.isValid() )
    {
      // set the flag for reversing the RecHits to false, since they are already in the correct order.
      trajectories.push_back(ReferenceTrajectoryPtr(new ReferenceTrajectory(input.first, input.second, false,
									    magneticField.product(), materialEffects(),
									    propagationDirection(), theMass, 
									    theUseBeamSpot, beamSpot)));
    }

    ++itTracks;
  }

  return trajectories;
}


const ReferenceTrajectoryFactory::ReferenceTrajectoryCollection
ReferenceTrajectoryFactory::trajectories( const edm::EventSetup & setup,
					  const ConstTrajTrackPairCollection& tracks,
					  const ExternalPredictionCollection& external,
					  const reco::BeamSpot &beamSpot) const
{
  ReferenceTrajectoryCollection trajectories;

  if ( tracks.size() != external.size() )
  {
    edm::LogInfo("ReferenceTrajectories") << "@SUB=ReferenceTrajectoryFactory::trajectories"
					  << "Inconsistent input:\n"
					  << "\tnumber of tracks = " << tracks.size()
					  << "\tnumber of external predictions = " << external.size();
    return trajectories;
  }

  edm::ESHandle< MagneticField > magneticField;
  setup.get< IdealMagneticFieldRecord >().get( magneticField );

  ConstTrajTrackPairCollection::const_iterator itTracks = tracks.begin();
  ExternalPredictionCollection::const_iterator itExternal = external.begin();

  while ( itTracks != tracks.end() )
  {
    TrajectoryInput input = innermostStateAndRecHits( *itTracks );
    // Check input: If all hits were rejected, the TSOS is initialized as invalid.
    if ( input.first.isValid() )
    {
      if ( (*itExternal).isValid() && sameSurface( (*itExternal).surface(), input.first.surface() ) )
      {
	// set the flag for reversing the RecHits to false, since they are already in the correct order.
	ReferenceTrajectoryPtr refTraj( new ReferenceTrajectory( *itExternal, input.second, false,
								 magneticField.product(), materialEffects(),
								 propagationDirection(), theMass,
								 theUseBeamSpot, beamSpot) );

	AlgebraicSymMatrix externalParamErrors( asHepMatrix<5>( (*itExternal).localError().matrix() ) );
	refTraj->setParameterErrors( externalParamErrors );

	trajectories.push_back( refTraj );
      }
      else
      {
	trajectories.push_back(ReferenceTrajectoryPtr(new ReferenceTrajectory(input.first, input.second, false,
									      magneticField.product(), materialEffects(),
									      propagationDirection(), theMass,
									      theUseBeamSpot, beamSpot)));
      }
    }

    ++itTracks;
    ++itExternal;
  }

  return trajectories;
}

DEFINE_EDM_PLUGIN( TrajectoryFactoryPlugin, ReferenceTrajectoryFactory, "ReferenceTrajectoryFactory" );
