
#include "Alignment/CommonAlignmentAlgorithm/interface/TrajectoryFactoryPlugin.h"


TrajectoryFactoryPlugin TrajectoryFactoryPlugin::theInstance;


TrajectoryFactoryPlugin::TrajectoryFactoryPlugin () : 
  seal::PluginFactory< TrajectoryFactoryBase *( const edm::ParameterSet & ) >( "TrajectoryFactoryPlugin" ) {}


TrajectoryFactoryPlugin* TrajectoryFactoryPlugin::get( void )
{
  return &theInstance;
}


TrajectoryFactoryBase* TrajectoryFactoryPlugin::getFactory( std::string factory,
							    const edm::ParameterSet & config )
{
  return theInstance.create( factory, config );
}
