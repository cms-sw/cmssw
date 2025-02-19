#ifndef DD_ALGO_PLUGIN_DD_ALGORITHM_FACTORY_H
# define DD_ALGO_PLUGIN_DD_ALGORITHM_FACTORY_H

//<<<<<< INCLUDES                                                       >>>>>>

#include "DetectorDescription/Algorithm/interface/DDAlgorithm.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

typedef edmplugin::PluginFactory<DDAlgorithm *(void)> DDAlgorithmFactory;

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // DD_ALGO_PLUGIN_DD_ALGORITHM_FACTORY_H
