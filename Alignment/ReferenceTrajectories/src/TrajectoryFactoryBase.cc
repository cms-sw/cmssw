#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"


TrajectoryFactoryBase::TrajectoryFactoryBase( const edm::ParameterSet & config ) : 
  theConfig(config)
{
  const std::string strMaterialEffects = config.getParameter< std::string >( "MaterialEffects" );
  theMaterialEffects = this->materialEffects( strMaterialEffects );
  
  const std::string strPropagationDirection = config.getParameter< std::string >( "PropagationDirection" );
  thePropDir = this->propagationDirection( strPropagationDirection );

  theUseWithoutDet = config.getParameter< bool >( "UseHitWithoutDet" );
  theUseInvalidHits = config.getParameter< bool >( "UseInvalidHits" );
  theUseProjectedHits = config.getParameter< bool >( "UseProjectedHits" );
  theUseBeamSpot = config.getParameter< bool >( "UseBeamSpot" );

  edm::LogInfo("Alignment") << "@SUB=TrajectoryFactoryBase"
                            << "TrajectoryFactory '" << config.getParameter<std::string>("TrajectoryFactoryName")
                            << "' with following settings:"
                            << "\nmaterial effects: " << strMaterialEffects
                            << "\npropagation: " << strPropagationDirection
                            << "\nuse hits without det: " << (theUseWithoutDet ? "yes" : "no")
                            << "\nuse invalid hits: " << (theUseInvalidHits ? "yes" : "no")
                            << "\nuse projected hits: " << (theUseProjectedHits ? "yes" : "no")
                            << "\nuse beamspot: " << (theUseBeamSpot ? "yes" : "no");
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


const Trajectory::DataContainer
TrajectoryFactoryBase::orderedTrajectoryMeasurements( const Trajectory & trajectory ) const
{
  const PropagationDirection dir = trajectory.direction();
  const bool hitsAreReverse = ( ( dir == thePropDir || thePropDir == anyDirection ) ? false : true );

  const Trajectory::DataContainer & original = trajectory.measurements();

  if ( hitsAreReverse )
  {
    // Simply use this line instead of the copying by hand?
    // const Trajectory::DataContainer reordered(original.rbegin(), original.rend());
    Trajectory::DataContainer reordered;
    reordered.reserve( original.size() );

    Trajectory::DataContainer::const_reverse_iterator itM;
    for ( itM = original.rbegin(); itM != original.rend(); ++itM )
    {
      reordered.push_back( *itM );
    }
    return reordered;
  }

  return original;
}


bool TrajectoryFactoryBase::sameSurface( const Surface& s1, const Surface& s2 ) const
{
  // - Should use perp2() instead of perp()
  // - Should not rely on floating point equality, but make a minimal range, e.g. 1.e-6 ?
  return ( s1.eta() == s2.eta() ) && ( s1.phi() == s2.phi() ) && ( s1.position().perp() == s2.position().perp() );
}


bool
TrajectoryFactoryBase::useRecHit( const TransientTrackingRecHit::ConstRecHitPointer& hitPtr ) const
{
  const GeomDet* det = hitPtr->det();
  if ( !det && !theUseWithoutDet ) return false;
  
  if ( !( theUseInvalidHits || hitPtr->isValid() ) ) return false;

  if ( !theUseProjectedHits )
  {
    const ProjectedRecHit2D* projectedHit = dynamic_cast< const ProjectedRecHit2D* >( hitPtr.get() );
    if ( projectedHit != 0 ) return false;
  }

  return true;
}


TrajectoryFactoryBase::MaterialEffects
TrajectoryFactoryBase::materialEffects( const std::string & strME ) const
{
  if ( strME == "MultipleScattering" ) return ReferenceTrajectoryBase::multipleScattering;
  if ( strME == "EnergyLoss" ) return ReferenceTrajectoryBase::energyLoss;
  if ( strME == "Combined" ) return ReferenceTrajectoryBase::combined;
  if ( strME == "None" ) return ReferenceTrajectoryBase::none;
  if ( strME == "BreakPoints" ) return ReferenceTrajectoryBase::breakPoints;
  if ( strME == "BrokenLines" ) return ReferenceTrajectoryBase::brokenLinesCoarse;
  if ( strME == "BrokenLinesCoarse" ) return ReferenceTrajectoryBase::brokenLinesCoarse;
  if ( strME == "BrokenLinesFine" ) return ReferenceTrajectoryBase::brokenLinesFine;
          
  throw cms::Exception("BadConfig")
    << "[TrajectoryFactoryBase::materialEffects] Unknown parameter: " << strME;
}


PropagationDirection
TrajectoryFactoryBase::propagationDirection( const std::string & strPD ) const
{
  if ( strPD == "oppositeToMomentum" ) return oppositeToMomentum;
  if ( strPD == "alongMomentum" ) return alongMomentum;
  if ( strPD == "anyDirection" ) return anyDirection;

  throw cms::Exception("BadConfig")
    << "[TrajectoryFactoryBase::propagationDirection] Unknown parameter: " << strPD;
}
