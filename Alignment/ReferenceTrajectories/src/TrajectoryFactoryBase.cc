#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"

#include <iostream>

TrajectoryFactoryBase::TrajectoryFactoryBase( const edm::ParameterSet & config )
{
  const std::string strMaterialEffects = config.getParameter< std::string >( "MaterialEffects" );
  theMaterialEffects = this->materialEffects( strMaterialEffects );

  const std::string strPropagationDirection = config.getParameter< std::string >( "PropagationDirection" );
  thePropDir = this->propagationDirection( strPropagationDirection );

  theUseInvalidHits = config.getParameter< bool >( "UseInvalidHits" );
}


TrajectoryFactoryBase::~TrajectoryFactoryBase( void ) {}


const TrajectoryFactoryBase::TrajectoryInput
TrajectoryFactoryBase::innermostStateAndRecHits( const ConstTrajTrackPair & track ) const
{
  TrajectoryFactoryBase::TrajectoryInput result; // pair of TSOS and ConstRecHitContainer

  // get the trajectory measurements in the correct order, i.e. reverse if needed
  Trajectory::DataContainer trajectoryMeasurements 
    = this->orderedTrajectoryMeasurements( *track.first );
  Trajectory::DataContainer::iterator itM = trajectoryMeasurements.begin();

  std::cout << "first measurement: r = " << trajectoryMeasurements.front().predictedState().globalPosition().perp() << std:: endl
	    << "last measurement: r = " << trajectoryMeasurements.back().predictedState().globalPosition().perp() << std:: endl;

  // get the innermost valid state
  while ( itM != trajectoryMeasurements.end() )
  {
    if ( ( *itM ).updatedState().isValid() ) break;
    ++itM;
  }
  if ( itM != trajectoryMeasurements.end() ) result.first = ( *itM ).updatedState();

  // get the valid RecHits
  while ( itM != trajectoryMeasurements.end() )
  {
    TransientTrackingRecHit::ConstRecHitPointer aRecHit = ( *itM ).recHit();
    if ( theUseInvalidHits || aRecHit->isValid() ) result.second.push_back( aRecHit );
    ++itM;
  }

  return result;
}


const Trajectory::DataContainer TrajectoryFactoryBase::orderedTrajectoryMeasurements( const Trajectory & trajectory ) const
{
  const PropagationDirection dir = trajectory.direction();
  const bool hitsAreReverse = ( dir == thePropDir ? false : true );

  const Trajectory::DataContainer & original = trajectory.measurements();

  if ( hitsAreReverse )
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


const TrajectoryFactoryBase::MaterialEffects TrajectoryFactoryBase::materialEffects( const std::string & strME ) const
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


const PropagationDirection TrajectoryFactoryBase::propagationDirection( const std::string & strPD ) const
{
  if ( strPD == "oppositeToMomentum" ) return oppositeToMomentum;
  if ( strPD == "alongMomentum" ) return alongMomentum;
  if ( strPD == "anyDirection" ) return anyDirection;

  edm::LogError("Alignment") << "@SUB=TrajectoryFactoryBase::propagationDirection" 
                             << "Unknown parameter \'" << strPD 
                             << "\'. I use \'anyDirection\' instead.";

  return anyDirection;
}
