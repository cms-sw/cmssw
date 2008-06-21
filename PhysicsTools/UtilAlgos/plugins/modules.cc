#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

DEFINE_SEAL_MODULE();

#include "PhysicsTools/CommonTools/interface/UpdaterService.h"
DEFINE_FWK_SERVICE( UpdaterService );

#include "PhysicsTools/CommonTools/interface/EventSelector.h"
#include "PhysicsTools/CommonTools/plugins/VariableEventSelector.h"
DEFINE_EDM_PLUGIN(EventSelectorFactory, VariableEventSelector, "VariableEventSelector");

#include "PhysicsTools/CommonTools/interface/CachingVariable.h"
DEFINE_EDM_PLUGIN(CachingVariableFactory, Power, "Power");
DEFINE_EDM_PLUGIN(CachingVariableFactory, VarSplitter, "VarSplitter");

#include "PhysicsTools/CommonTools/interface/Plotter.h"
DEFINE_EDM_PLUGIN(PlotterFactory, VariablePlotter, "VariablePlotter");
