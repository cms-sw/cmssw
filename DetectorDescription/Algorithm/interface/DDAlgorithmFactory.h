#ifndef DD_ALGO_PLUGIN_DD_ALGORITHM_FACTORY_H
# define DD_ALGO_PLUGIN_DD_ALGORITHM_FACTORY_H

//<<<<<< INCLUDES                                                       >>>>>>

#include "DetectorDescription/DDAlgorithm/interface/DDAlgorithm.h"
#include "PluginManager/PluginFactory.h"

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

class DDAlgorithmFactory : public seal::PluginFactory<DDAlgorithm *(void)>
{
public:
    static DDAlgorithmFactory *get (void);

private:
    DDAlgorithmFactory (void);
    static DDAlgorithmFactory s_instance;
};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // DD_ALGO_PLUGIN_DD_ALGORITHM_FACTORY_H
