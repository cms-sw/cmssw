#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"


TrajectoryFactoryBase::TrajectoryFactoryBase(const edm::ParameterSet& config) :
  cfg_(config),
  materialEffects_(materialEffects(config.getParameter<std::string>("MaterialEffects"))),
  propDir_(propagationDirection(config.getParameter<std::string>("PropagationDirection"))),
  useWithoutDet_(config.getParameter<bool>("UseHitWithoutDet")),
  useInvalidHits_(config.getParameter<bool>("UseInvalidHits")),
  useProjectedHits_(config.getParameter<bool>("UseProjectedHits")),
  useBeamSpot_(config.getParameter<bool>("UseBeamSpot")),
  includeAPEs_(config.getParameter<bool>("IncludeAPEs"))
{
  edm::LogInfo("Alignment")
    << "@SUB=TrajectoryFactoryBase"
    << "TrajectoryFactory '" << cfg_.getParameter<std::string>("TrajectoryFactoryName")
    << "' with following settings:"
    << "\nmaterial effects: " << cfg_.getParameter<std::string>("MaterialEffects")
    << "\npropagation: " << cfg_.getParameter<std::string>("PropagationDirection")
    << "\nuse hits without det: " << (useWithoutDet_ ? "yes" : "no")
    << "\nuse invalid hits: " << (useInvalidHits_ ? "yes" : "no")
    << "\nuse projected hits: " << (useProjectedHits_ ? "yes" : "no")
    << "\nuse beamspot: " << (useBeamSpot_ ? "yes" : "no")
    << "\ninclude APEs: " << (includeAPEs_ ? "yes" : "no");
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
  const bool hitsAreReverse = ( ( dir == propDir_ || propDir_ == anyDirection ) ? false : true );

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
  if ( !det && !useWithoutDet_ ) return false;
  
  if ( !( useInvalidHits_ || hitPtr->isValid() ) ) return false;

  if ( !useProjectedHits_ )
  {
    if(trackerHitRTTI::isProjected(*hitPtr)) return false;
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
  if ( strME == "LocalGBL" ) return ReferenceTrajectoryBase::localGBL;
  if ( strME == "CurvlinGBL" ) return ReferenceTrajectoryBase::curvlinGBL;
            
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
