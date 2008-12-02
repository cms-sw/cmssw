#ifndef Fireworks_Core_FWGlimpseDataProxyBuilderFactory_h
#define Fireworks_Core_FWGlimpseDataProxyBuilderFactory_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGlimpseDataProxyBuilderFactory
//
/**\class FWGlimpseDataProxyBuilderFactory FWGlimpseDataProxyBuilderFactory.h Fireworks/Core/interface/FWGlimpseDataProxyBuilderFactory.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Jun  5 20:13:55 EDT 2008
// $Id: FWGlimpseDataProxyBuilderFactory.h,v 1.2 2008/11/06 22:05:23 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/register_dataproxybuilder_macro.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

// forward declarations

class FWGlimpseDataProxyBuilder;

typedef edmplugin::PluginFactory<FWGlimpseDataProxyBuilder*()> FWGlimpseDataProxyBuilderFactory;

#define REGISTER_FWGLIMPSEDATAPROXYBUILDER(_name_,_type_,_purpose_) \
DEFINE_PROXYBUILDER_METHODS(_name_,_type_,_purpose_); \
DEFINE_EDM_PLUGIN(FWGlimpseDataProxyBuilderFactory,_name_,_name_::typeOfBuilder()+_name_::classTypeName()+"@"+_name_::classPurpose()+"@" #_name_)

#endif
