#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"


TrajectoryFactoryBase::TrajectoryFactoryBase( const edm::ParameterSet & config )
{
  const std::string strMaterialEffects = config.getParameter< std::string >( "MaterialEffects" );
  theMaterialEffects = this->materialEffects( strMaterialEffects );

  const std::string strPropagationDirection = config.getParameter< std::string >( "PropagationDirection" );
  thePropDir = this->propagationDirection( strPropagationDirection );

  theUseInvalidHits = config.getParameter< bool >( "UseInvalidHits" );
  theUseProjectedHits = config.getParameter< bool >( "UseProjectedHits" );
}


TrajectoryFactoryBase::~TrajectoryFactoryBase( void ) {}



const TrajectoryFactoryBase::TrajectoryInput
TrajectoryFactoryBase::innermostStateAndRecHits( const ConstTrajTrackPair & track ) const
{
  TrajectoryInput result;

  // get the trajectory measurements in the correct order, i.e. reverse if needed
  Trajectory::DataContainer trajectoryMeasurements 
    = this->orderedTrajectoryMeasurements( *track.first );
  Trajectory::DataContainer::iterator itM = trajectoryMeasurements.begin();

  // get the innermost valid trajectory state - the corresponding hit must be o.k. as well
  while ( itM != trajectoryMeasurements.end() )
  {
    if ( ( *itM ).updatedState().isValid() && useRecHit( ( *itM ).recHit() ) ) break;
    ++itM;
  }
  if ( itM != trajectoryMeasurements.end() ) result.first = ( *itM ).updatedState();

  // get the valid RecHits
  while ( itM != trajectoryMeasurements.end() )
  {
    TransientTrackingRecHit::ConstRecHitPointer aRecHit = ( *itM ).recHit();
    if ( useRecHit( aRecHit ) ) result.second.push_back( aRecHit );
    ++itM;
  }

  return result;
}


const Trajectory::DataContainer TrajectoryFactoryBase::orderedTrajectoryMeasurements( const Trajectory & trajectory ) const
{
  const PropagationDirection dir = trajectory.direction();
  const bool hitsAreReverse = ( ( dir == thePropDir || thePropDir == anyDirection ) ? false : true );

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


bool TrajectoryFactoryBase::sameSurface( const Surface& s1, const Surface& s2 ) const
{
  return ( s1.eta() == s2.eta() ) && ( s1.phi() == s2.phi() ) && ( s1.position().perp() == s2.position().perp() );
}


bool TrajectoryFactoryBase::useRecHit( const TransientTrackingRecHit::ConstRecHitPointer& hitPtr ) const
{
  bool useHit = true;

  if ( !( theUseInvalidHits || hitPtr->isValid() ) ) useHit = false;

  if ( !theUseProjectedHits )
  {
    const ProjectedRecHit2D* projectedHit = dynamic_cast< const ProjectedRecHit2D* >( hitPtr.get() );
    if ( projectedHit != 0 ) useHit = false;
  }

  return useHit;
}


const TrajectoryFactoryBase::MaterialEffects TrajectoryFactoryBase::materialEffects( const std::string & strME ) const
{
  if ( strME == "MultipleScattering" ) return ReferenceTrajectoryBase::multipleScattering;
  if ( strME == "EnergyLoss" ) return ReferenceTrajectoryBase::energyLoss;
  if ( strME == "Combined" ) return ReferenceTrajectoryBase::combined;
  if ( strME == "None" ) return ReferenceTrajectoryBase::none;

  throw cms::Exception("BadConfig")
    << "[TrajectoryFactoryBase::materialEffects] Unknown parameter: " << strME;
}


const PropagationDirection TrajectoryFactoryBase::propagationDirection( const std::string & strPD ) const
{
  if ( strPD == "oppositeToMomentum" ) return oppositeToMomentum;
  if ( strPD == "alongMomentum" ) return alongMomentum;
  if ( strPD == "anyDirection" ) return anyDirection;

  throw cms::Exception("BadConfig")
    << "[TrajectoryFactoryBase::propagationDirection] Unknown parameter: " << strPD;
}
