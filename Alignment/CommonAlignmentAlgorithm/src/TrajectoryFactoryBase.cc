#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/TrajectoryFactoryBase.h"

TrajectoryFactoryBase::TrajectoryFactoryBase( const edm::ParameterSet & config )
{
  theHitsAreReverse = config.getParameter< bool >( "ReverseHits" );
  std::string strMaterialEffects = config.getParameter< std::string >( "MaterialEffects" );
  theMaterialEffects = materialEffects( strMaterialEffects );
}


TrajectoryFactoryBase::~TrajectoryFactoryBase( void ) {}


const TrajectoryFactoryBase::MaterialEffects TrajectoryFactoryBase::materialEffects( const std::string strME ) const
{
  if ( strME == "MultipleScattering" ) return ReferenceTrajectoryBase::multipleScattering;
  if ( strME == "EnergyLoss" ) return ReferenceTrajectoryBase::energyLoss;
  if ( strME == "Combined" ) return ReferenceTrajectoryBase::combined;
  if ( strME == "None" ) return ReferenceTrajectoryBase::none;

  edm::LogError("Alignment") << "@SUB=TrajectoryFactoryBase::materialEffects" 
                             << "Unknown parameter \'" << strME 
                             << "\'. I use \'Combined\' instead.";

  return ReferenceTrajectoryBase::combined;
}


const TrajectoryFactoryBase::TrajectoryInput
TrajectoryFactoryBase::innermostStateAndRecHits( const ConstTrajTrackPair & track ) const
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


const Trajectory::DataContainer TrajectoryFactoryBase::orderedTrajectoryMeasurements( const Trajectory & trajectory ) const
{
  const Trajectory::DataContainer & original = trajectory.measurements();

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
