#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Pythia8/Pythia.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

// Automatic addition of user hooks to pythia without the need to edit the hadronizer/generator files
typedef edmplugin::PluginFactory<Pythia8::UserHooks*(const edm::ParameterSet&)> CustomHookFactory;

#define REGISTER_USERHOOK(type) DEFINE_EDM_PLUGIN(CustomHookFactory, type, #type)
