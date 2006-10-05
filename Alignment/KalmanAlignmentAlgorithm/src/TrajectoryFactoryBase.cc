
#include "Alignment/KalmanAlignmentAlgorithm/interface/TrajectoryFactoryBase.h"

#include <iostream>


TrajectoryFactoryBase::TrajectoryFactoryBase( const edm::ParameterSet & config ) {}


TrajectoryFactoryBase::~TrajectoryFactoryBase( void ) {}


const TrajectoryFactoryBase::MaterialEffects TrajectoryFactoryBase::materialEffects( const std::string strME ) const
{
  if ( strME == "MultipleScattering" ) return ReferenceTrajectoryBase::multipleScattering;
  if ( strME == "EnergyLoss" ) return ReferenceTrajectoryBase::energyLoss;
  if ( strME == "Combined" ) return ReferenceTrajectoryBase::combined;
  if ( strME == "None" ) return ReferenceTrajectoryBase::none;

  std::cout << "[TrajectoryFactoryBase::materialEffects] Unknown parameter \'" 
	    << strME << "\'. I use \'Combined\' instead." << std::endl;
  return ReferenceTrajectoryBase::combined;
}



