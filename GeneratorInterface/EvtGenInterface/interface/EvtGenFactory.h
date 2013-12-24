#ifndef GeneratorInterface_EvtGenInterface_EvtGenFactory_H
#define GeneratorInterface_EvtGenInterface_EvtGenFactory_H
 
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "GeneratorInterface/EvtGenInterface/interface/EvtGenInterface.h"

typedef edmplugin::PluginFactory<gen::EvtGenInterfaceBase* (const edm::ParameterSet&) >  EvtGenFactory;

#endif
