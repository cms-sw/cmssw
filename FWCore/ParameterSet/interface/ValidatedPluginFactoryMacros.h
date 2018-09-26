#ifndef FWCore_ParameterSet_ValidatedPluginFactoryMacros_h
#define FWCore_ParameterSet_ValidatedPluginFactoryMacros_h
// -*- C++ -*-
//
// Package:     FWCore/ParameterSet
// Class  :     ValidatedPluginFactoryMacros
// 
/**\class ValidatedPluginFactoryMacros ValidatedPluginFactoryMacros.h "ValidatedPluginFactoryMacros.h"

 Description: macros used to register a PluginFactory which uses ParameterSet validation

 Usage:
    Call EDM_REGISTER_VALIDATED_PLUGINFACTORY instead of EDM_REGISTER_PLUGINFACTORY for the case
 where you want to use edm::PluginDescription to validate the plugins. Make sure to also use
 DEFINE_EDM_VALIDATED_PLUGIN instead of DEFINE_EDM_PLUGIN when registering the plugins for this
 factory.

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 21 Sep 2018 13:14:22 GMT
//

// system include files

// user include files
#include "FWCore/ParameterSet/interface/PluginDescriptionAdaptorBase.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

#define EDM_REGISTER_VALIDATED_PLUGINFACTORY(_factory_,_category_) \
EDM_REGISTER_PLUGINFACTORY(_factory_,_category_);\
EDM_REGISTER_PLUGINFACTORY2(edmplugin::PluginFactory<edm::PluginDescriptionAdaptorBase<_factory_::CreatedType>*()>, "PluginDescriptor" _category_)
#endif
