#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

DEFINE_SEAL_MODULE();

#include "PhysicsTools/UtilAlgos/interface/EventSelector.h"
#include "PhysicsTools/UtilAlgos/plugins/VariableEventSelector.h"
DEFINE_EDM_PLUGIN(EventSelectorFactory, VariableEventSelector, "VariableEventSelector");

#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"
DEFINE_EDM_PLUGIN(CachingVariableFactory, Power, "Power");
DEFINE_EDM_PLUGIN(CachingVariableFactory, VarSplitter, "VarSplitter");

#include "PhysicsTools/UtilAlgos/interface/Plotter.h"
DEFINE_EDM_PLUGIN(PlotterFactory, VariablePlotter, "VariablePlotter");
