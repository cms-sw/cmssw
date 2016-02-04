#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentUpdatorPlugin_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentUpdatorPlugin_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdator.h"

/// A PluginFactory for updators for the KalmanAlignmentAlgorithm.


// Forward declaration
namespace edm { class ParameterSet; }

typedef edmplugin::PluginFactory<KalmanAlignmentUpdator* ( const edm::ParameterSet& ) >
                   KalmanAlignmentUpdatorPlugin;

 
#endif
