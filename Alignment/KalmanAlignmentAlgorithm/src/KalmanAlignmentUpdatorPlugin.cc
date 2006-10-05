
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdatorPlugin.h"


KalmanAlignmentUpdatorPlugin KalmanAlignmentUpdatorPlugin::theInstance;


KalmanAlignmentUpdatorPlugin::KalmanAlignmentUpdatorPlugin () : 
  seal::PluginFactory< KalmanAlignmentUpdator *( const edm::ParameterSet & ) >( "KalmanAlignmentUpdatorPlugin" ) {}


KalmanAlignmentUpdatorPlugin* KalmanAlignmentUpdatorPlugin::get( void )
{
  return &theInstance;
}


KalmanAlignmentUpdator* KalmanAlignmentUpdatorPlugin::getUpdator( std::string updator,
								  const edm::ParameterSet & config )
{
  return theInstance.create( updator, config );
}
