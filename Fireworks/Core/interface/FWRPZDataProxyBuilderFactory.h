#ifndef Fireworks_Core_FWRPZDataProxyBuilderFactory_h
#define Fireworks_Core_FWRPZDataProxyBuilderFactory_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZDataProxyBuilderFactory
//
/**\class FWRPZDataProxyBuilderFactory FWRPZDataProxyBuilderFactory.h Fireworks/Core/interface/FWRPZDataProxyBuilderFactory.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Jun  5 20:13:37 EDT 2008
// $Id: FWRPZDataProxyBuilderFactory.h,v 1.1 2008/06/09 19:48:44 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/register_dataproxybuilder_macro.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

// forward declarations
class FWRPZDataProxyBuilder;

typedef edmplugin::PluginFactory<FWRPZDataProxyBuilder*()> FWRPZDataProxyBuilderFactory;

#define REGISTER_FWRPZDATAPROXYBUILDER(_name_,_type_,_purpose_) \
DEFINE_PROXYBUILDER_METHODS(_name_,_type_,_purpose_); \
DEFINE_EDM_PLUGIN(FWRPZDataProxyBuilderFactory,_name_,_name_::classTypeName()+"@"+_name_::classPurpose()+"@" #_name_)

#endif
