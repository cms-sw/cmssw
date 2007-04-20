#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentMetricsUpdatorPlugin_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentMetricsUpdatorPlugin_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdator.h"

/// A PluginFactory for concrete instances of class KalmanAlignmentMetricsUpdator.

// Forward declaration
namespace edm { class ParameterSet; }

typedef edmplugin::PluginFactory<KalmanAlignmentMetricsUpdator* ( const edm::ParameterSet& ) >
                   KalmanAlignmentMetricsUpdatorPlugin;

 

#endif
