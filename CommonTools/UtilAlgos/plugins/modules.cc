#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

DEFINE_SEAL_MODULE();

#include "CommonTools/UtilAlgos/interface/VariableHelper.h"
DEFINE_FWK_SERVICE( VariableHelperService );
#include "CommonTools/UtilAlgos/interface/InputTagDistributor.h"
DEFINE_FWK_SERVICE( InputTagDistributorService );

#include "CommonTools/UtilAlgos/interface/EventSelector.h"
#include "CommonTools/UtilAlgos/plugins/VariableEventSelector.h"
DEFINE_EDM_PLUGIN(EventSelectorFactory, VariableEventSelector, "VariableEventSelector");

#include "CommonTools/UtilAlgos/interface/CachingVariable.h"
DEFINE_EDM_PLUGIN(CachingVariableFactory, VariablePower, "VariablePower");
DEFINE_EDM_PLUGIN(CachingVariableFactory, VarSplitter, "VarSplitter");

typedef SimpleValueVariable<double> DoubleVar;
typedef SimpleValueVariable<bool> BoolVar;
typedef SimpleValueVectorVariable<bool> DoubleVVar;
typedef SimpleValueVectorVariable<bool> BoolVVar;
DEFINE_EDM_PLUGIN(CachingVariableFactory, DoubleVar, "DoubleVar");
DEFINE_EDM_PLUGIN(CachingVariableFactory, BoolVar, "BoolVar");
DEFINE_EDM_PLUGIN(CachingVariableFactory, DoubleVVar, "DoubleVVar");
DEFINE_EDM_PLUGIN(CachingVariableFactory, BoolVVar, "BoolVVar");

DEFINE_EDM_PLUGIN(CachingVariableFactory, ComputedVariable, "ComputedVariable");

DEFINE_EDM_PLUGIN(VariableComputerFactory, VariableComputerTest, "VariableComputerTest");

#include "CommonTools/UtilAlgos/interface/Plotter.h"
DEFINE_EDM_PLUGIN(PlotterFactory, VariablePlotter, "VariablePlotter");
