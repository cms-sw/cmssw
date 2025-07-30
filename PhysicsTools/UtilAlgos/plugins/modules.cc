#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "CommonTools/UtilAlgos/interface/EventSelector.h"
#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"

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

#include "PhysicsTools/UtilAlgos/interface/StringBasedNTupler.h"
DEFINE_EDM_PLUGIN(NTuplerFactory, StringBasedNTupler, "StringBasedNTupler");
