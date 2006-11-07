#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/TrajectoryFactoryBase.h"

TrajectoryFactoryBase::TrajectoryFactoryBase( const edm::ParameterSet & config ) {}


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



