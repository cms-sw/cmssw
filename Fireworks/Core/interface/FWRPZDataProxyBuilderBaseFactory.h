#ifndef Fireworks_Core_FWRPZDataProxyBuilderBaseFactory_h
#define Fireworks_Core_FWRPZDataProxyBuilderBaseFactory_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZDataProxyBuilderBaseFactory
//
/**\class FWRPZDataProxyBuilderBaseFactory FWRPZDataProxyBuilderBaseFactory.h Fireworks/Core/interface/FWRPZDataProxyBuilderBaseFactory.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Jun  5 20:13:37 EDT 2008
// $Id: FWRPZDataProxyBuilderBaseFactory.h,v 1.1 2008/11/26 01:50:46 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/register_dataproxybuilder_macro.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

// forward declarations
class FWRPZDataProxyBuilderBase;

typedef edmplugin::PluginFactory<FWRPZDataProxyBuilderBase*()> FWRPZDataProxyBuilderBaseFactory;

#define REGISTER_FWRPZDATAPROXYBUILDERBASE(_name_,_type_,_purpose_) \
   DEFINE_PROXYBUILDER_METHODS(_name_,_type_,_purpose_); \
   DEFINE_EDM_PLUGIN(FWRPZDataProxyBuilderBaseFactory,_name_,_name_::typeOfBuilder()+_name_::classTypeName()+"@"+_name_::classPurpose()+"@" # _name_)

#endif
