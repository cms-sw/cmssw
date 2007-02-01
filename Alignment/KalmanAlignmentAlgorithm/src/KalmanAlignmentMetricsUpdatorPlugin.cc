
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdatorPlugin.h"


KalmanAlignmentMetricsUpdatorPlugin KalmanAlignmentMetricsUpdatorPlugin::theInstance;


KalmanAlignmentMetricsUpdatorPlugin::KalmanAlignmentMetricsUpdatorPlugin () : 
  seal::PluginFactory< KalmanAlignmentMetricsUpdator *( const edm::ParameterSet & ) >( "KalmanAlignmentMetricsUpdatorPlugin" ) {}


KalmanAlignmentMetricsUpdatorPlugin* KalmanAlignmentMetricsUpdatorPlugin::get( void )
{
  return &theInstance;
}


KalmanAlignmentMetricsUpdator* KalmanAlignmentMetricsUpdatorPlugin::getUpdator( std::string updator,
										const edm::ParameterSet & config )
{
  return theInstance.create( updator, config );
}
