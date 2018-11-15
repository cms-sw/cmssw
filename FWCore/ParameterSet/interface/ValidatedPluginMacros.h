#ifndef FWCore_ParameterSet_ValidatedPluginMacros_h
#define FWCore_ParameterSet_ValidatedPluginMacros_h
// -*- C++ -*-
//
// Package:     FWCore/ParameterSet
// Class  :     ValidatedPluginMacros
// 
/**\class ValidatedPluginMacros ValidatedPluginMacros.h "ValidatedPluginMacros.h"

 Description: Registration macros for plugins that use ParameterSet validation

 Usage:
 Call DEFINE_EDM_VALIDATED_PLUGIN instead of DEFINE_EDM_PLUGIN for the case
 where you registered the PluginFactory with ED_REGISTER_VALIDATED_PLUGINFACTORY.

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 21 Sep 2018 13:09:38 GMT
//

// system include files

// user include files
#include "FWCore/ParameterSet/interface/PluginDescriptionAdaptor.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

#define DEFINE_EDM_VALIDATED_PLUGIN(factory, type, name) \
DEFINE_EDM_PLUGIN(factory, type, name); \
using EDM_PLUGIN_SYM(adaptor_t,__LINE__) = edm::PluginDescriptionAdaptor<factory::CreatedType,type>; \
DEFINE_EDM_PLUGIN2(edmplugin::PluginFactory<edm::PluginDescriptionAdaptorBase<factory::CreatedType>*()>, EDM_PLUGIN_SYM(adaptor_t,__LINE__), name)

#endif
