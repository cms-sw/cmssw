#ifndef FASTSIM_INTERACTIONMODELFACTORY
#define FASTSIM_INTERACTIONMODELFACTORY

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "string"

namespace edm
{
    class ParameterSet;
}

namespace fastsim
{
    class InteractionModel;
    typedef edmplugin::PluginFactory<fastsim::InteractionModel*(const std::string & name,const edm::ParameterSet&)> InteractionModelFactory;
}


#endif
