#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"



#include "PhysicsTools/UtilAlgos/interface/VariableHelper.h"
DEFINE_FWK_SERVICE( VariableHelperService );
#include "PhysicsTools/UtilAlgos/interface/InputTagDistributor.h"
DEFINE_FWK_SERVICE( InputTagDistributorService );

#include "CommonTools/UtilAlgos/interface/EventSelector.h"
#include "PhysicsTools/UtilAlgos/plugins/VariableEventSelector.h"
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, VariableEventSelector, "VariableEventSelector");
DEFINE_EDM_PLUGIN(EventSelectorFactoryFromHelper, VariableFormulaEventSelector, "VariableFormulaEventSelector");

#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"
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

#include "PhysicsTools/UtilAlgos/interface/Plotter.h"
DEFINE_EDM_PLUGIN(PlotterFactory, VariablePlotter, "VariablePlotter");


#include "PhysicsTools/UtilAlgos/interface/StringBasedNTupler.h"
DEFINE_EDM_PLUGIN(NTuplerFactory, StringBasedNTupler, "StringBasedNTupler");
#include "PhysicsTools/UtilAlgos/interface/VariableNTupler.h"
DEFINE_EDM_PLUGIN(NTuplerFactory, VariableNTupler, "VariableNTupler");
#include "PhysicsTools/UtilAlgos/interface/CompleteNTupler.h"
DEFINE_EDM_PLUGIN(NTuplerFactory, CompleteNTupler, "CompleteNTupler");
