#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentMetricsUpdatorPlugin_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentMetricsUpdatorPlugin_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdator.h"

#include <string>

/// A PluginFactory for concrete instances of class KalmanAlignmentMetricsUpdator.

 
class KalmanAlignmentMetricsUpdatorPlugin : public seal::PluginFactory< KalmanAlignmentMetricsUpdator *( const edm::ParameterSet & ) >
{

public:

  KalmanAlignmentMetricsUpdatorPlugin( void );

  static KalmanAlignmentMetricsUpdatorPlugin* get( void );

  static KalmanAlignmentMetricsUpdator* getUpdator( std::string updator, const edm::ParameterSet & config );

private:

  static KalmanAlignmentMetricsUpdatorPlugin theInstance;

};

#endif
