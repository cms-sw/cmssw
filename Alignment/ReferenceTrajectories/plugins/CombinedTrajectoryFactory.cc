// Do not include .h from plugin directory, but locally:
#include "CombinedTrajectoryFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"

using namespace std;


CombinedTrajectoryFactory::CombinedTrajectoryFactory( const edm::ParameterSet & config ) :
  TrajectoryFactoryBase( config )
{
  vector<string> factoryNames = config.getParameter< vector<string> >( "TrajectoryFactoryNames" );
  vector<string>::iterator itFactoryName;
  for ( itFactoryName = factoryNames.begin(); itFactoryName != factoryNames.end(); ++itFactoryName )
  {
    theFactories.push_back( TrajectoryFactoryPlugin::get()->create( *itFactoryName, config ) );
  }
}

 
CombinedTrajectoryFactory::~CombinedTrajectoryFactory( void ) {}


const CombinedTrajectoryFactory::ReferenceTrajectoryCollection
CombinedTrajectoryFactory::trajectories( const edm::EventSetup & setup,
					 const ConstTrajTrackPairCollection & tracks ) const
{
  ReferenceTrajectoryCollection trajectories;

  vector< TrajectoryFactoryBase* >::const_iterator itFactory;
  for ( itFactory = theFactories.begin(); trajectories.empty() && ( itFactory != theFactories.end() ); ++itFactory )
  {
    trajectories = ( *itFactory )->trajectories( setup, tracks );
  }

  return trajectories;
}

const CombinedTrajectoryFactory::ReferenceTrajectoryCollection
CombinedTrajectoryFactory::trajectories( const edm::EventSetup & setup,
					 const ConstTrajTrackPairCollection& tracks,
					 const ExternalPredictionCollection& external ) const
{
  ReferenceTrajectoryCollection trajectories;

  vector< TrajectoryFactoryBase* >::const_iterator itFactory;
  for ( itFactory = theFactories.begin(); trajectories.empty() && ( itFactory != theFactories.end() ); ++itFactory )
  {
    trajectories = ( *itFactory )->trajectories( setup, tracks, external );
  }

  return trajectories;
}


DEFINE_EDM_PLUGIN( TrajectoryFactoryPlugin, CombinedTrajectoryFactory, "CombinedTrajectoryFactory" );
