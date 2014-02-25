#ifndef GeneratorInterface_TauolaInterface_TauolaFactory_H
#define GeneratorInterface_TauolaInterface_TauolaFactory_H
 
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaInterfaceBase.h"

typedef edmplugin::PluginFactory<gen::TauolaInterfaceBase* (const edm::ParameterSet&) >  TauolaFactory;

#endif
