
#include "Alignment/KalmanAlignmentAlgorithm/interface/ReferenceTrajectoryFactory.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include <iostream>


ReferenceTrajectoryFactory::ReferenceTrajectoryFactory( const edm::ParameterSet & config ) :
  TrajectoryFactoryBase( config )
{
  theHitsAreReverse = config.getParameter< bool >( "ReverseHits" );
  std::string strMaterialEffects = config.getParameter< std::string >( "MaterialEffects" );
  theMaterialEffects = materialEffects( strMaterialEffects );
  theMass = config.getParameter< double >( "ParticleMass" );
}

 
ReferenceTrajectoryFactory::~ReferenceTrajectoryFactory( void ) {}


const ReferenceTrajectoryFactory::ReferenceTrajectoryCollection
ReferenceTrajectoryFactory::trajectories( const edm::EventSetup & setup,
				       const TrajTrackPairCollection & tracks ) const
{
  ReferenceTrajectoryCollection trajectories;

  edm::ESHandle< MagneticField > magneticField;
  setup.get< IdealMagneticFieldRecord >().get( magneticField );

  TrajTrackPairCollection::const_iterator itTracks = tracks.begin();

  while ( itTracks != tracks.end() )
  { 
    TrajectoryInput input = innermostStateAndRecHits( *itTracks );
    // set the flag for reversing the RecHits to false, since they are already in the correct order.
    trajectories.push_back( ReferenceTrajectoryPtr( new ReferenceTrajectory( input.first, input.second, 
									     false, magneticField.product(),
									     theMaterialEffects, theMass ) ) );
    ++itTracks;
  }

  return trajectories;
}


const ReferenceTrajectoryFactory::TrajectoryInput
ReferenceTrajectoryFactory::innermostStateAndRecHits( const TrajTrackPair & track ) const
{
  TransientTrackingRecHit::ConstRecHitContainer recHits;
  TrajectoryStateOnSurface innermostState;

  // get the trajectory measurements in the correct order, i.e. reverse if needed
  Trajectory::DataContainer trajectoryMeasurements = orderedTrajectoryMeasurements( *track.first );
  Trajectory::DataContainer::iterator itM = trajectoryMeasurements.begin();

  // get the innermost valid state
  while ( itM != trajectoryMeasurements.end() )
  {
    if ( ( *itM ).updatedState().isValid() ) break;
    ++itM;
  }
  if ( itM != trajectoryMeasurements.end() ) innermostState = ( *itM ).updatedState();

  // get the valid RecHits
  while ( itM != trajectoryMeasurements.end() )
  {
    TransientTrackingRecHit::ConstRecHitPointer aRecHit = ( *itM ).recHit();
    if ( aRecHit->isValid() ) recHits.push_back( aRecHit );
    ++itM;
  }

  return make_pair( innermostState, recHits );
}


const Trajectory::DataContainer ReferenceTrajectoryFactory::orderedTrajectoryMeasurements( const Trajectory & trajectory ) const
{
  const Trajectory::DataContainer original = trajectory.measurements();

  if ( theHitsAreReverse )
  {
    Trajectory::DataContainer reordered;
    reordered.reserve( original.size() );
    for ( Trajectory::DataContainer::const_reverse_iterator itM = original.rbegin(); itM != original.rend(); ++itM )
    {
      reordered.push_back( *itM );
    }
    return reordered;
  }

  return original;
}
