#ifndef Fireworks_Core_FWProxyBuilderFactory_h
#define Fireworks_Core_FWProxyBuilderFactory_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWProxyBuilderFactory
//
/**\class FWProxyBuilderFactory FWProxyBuilderFactory.h Fireworks/Core/interface/FWProxyBuilderFactory.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Jun  5 20:13:55 EDT 2008
// $Id: FWProxyBuilderFactory.h,v 1.2 2010/06/02 22:40:33 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/register_dataproxybuilder_macro.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

// forward declarations

class FWProxyBuilderBase;

typedef edmplugin::PluginFactory<FWProxyBuilderBase*()> FWProxyBuilderFactory;

#define REGISTER_FWPROXYBUILDER(_name_,_type_,_purpose_,_view_) \
   DEFINE_PROXYBUILDER_METHODS(_name_,_type_,_purpose_,_view_); \
   DEFINE_EDM_PLUGIN(FWProxyBuilderFactory,_name_,_name_::typeOfBuilder()+_name_::classRegisterTypeName()+(_name_::representsSubPart()?"!":"_")+"@"+_name_::classPurpose()+"@"+_name_::classView()+"#" # _name_)


#endif
