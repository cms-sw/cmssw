#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentUpdatorPlugin_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentUpdatorPlugin_h

#include "PluginManager/PluginFactory.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdator.h"

#include <string>

/// A PluginFactory for updators for the KalmanAlignmentAlgorithm.

 
class KalmanAlignmentUpdatorPlugin : public seal::PluginFactory< KalmanAlignmentUpdator *( const edm::ParameterSet & ) >
{

public:

  KalmanAlignmentUpdatorPlugin( void );

  static KalmanAlignmentUpdatorPlugin* get( void );

  static KalmanAlignmentUpdator* getUpdator( std::string updator, const edm::ParameterSet & config );

private:

  static KalmanAlignmentUpdatorPlugin theInstance;

};

#endif
