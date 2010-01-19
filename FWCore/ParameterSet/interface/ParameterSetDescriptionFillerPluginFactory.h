#ifndef FWCore_ParameterSet_ParameterSetDescriptionFillerPluginFactory_h
#define FWCore_ParameterSet_ParameterSetDescriptionFillerPluginFactory_h
// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterSetDescriptionFillerPluginFactory
// 
/**\class ParameterSetDescriptionFillerPluginFactory ParameterSetDescriptionFillerPluginFactory.h FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h

 Description: Provides access to the ParameterSetDescription object of a plugin

 Usage:
    The Factory allows loading of plugins and querying them for their ParameterSetDescription.

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Aug  1 16:47:01 EDT 2007
//

// system include files

// user include files
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"

// forward declarations

namespace edm {
  typedef edmplugin::PluginFactory<ParameterSetDescriptionFillerBase*(void)> ParameterSetDescriptionFillerPluginFactory;
}

#define DEFINE_FWK_PSET_DESC_FILLER(type) \
static edm::ParameterSetDescriptionFillerPluginFactory::PMaker<edm::ParameterSetDescriptionFiller<type > > EDM_PLUGIN_SYM(s_filler , __LINE__ ) (#type)
//NOTE: Can't do the below since this appears on the same line as another factory and w'ed have two variables with the same name
//DEFINE_EDM_PLUGIN (edm::ParameterSetDescriptionFillerPluginFactory,type,#type)

// Define another analogous macro to handle the special case of services.

#define DEFINE_DESC_FILLER_FOR_SERVICES(pluginName, serviceType)		\
static edm::ParameterSetDescriptionFillerPluginFactory::PMaker<edm::DescriptionFillerForServices<serviceType > > EDM_PLUGIN_SYM(s_filler , __LINE__ ) (#pluginName)

#endif
