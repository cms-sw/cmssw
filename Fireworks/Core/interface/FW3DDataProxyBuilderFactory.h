#ifndef Fireworks_Core_FW3DDataProxyBuilderFactory_h
#define Fireworks_Core_FW3DDataProxyBuilderFactory_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DDataProxyBuilderFactory
//
/**\class FW3DDataProxyBuilderFactory FW3DDataProxyBuilderFactory.h Fireworks/Core/interface/FW3DDataProxyBuilderFactory.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Jun  5 20:13:55 EDT 2008
// $Id: FW3DDataProxyBuilderFactory.h,v 1.1 2008/12/01 12:27:36 dmytro Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/register_dataproxybuilder_macro.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

// forward declarations

class FW3DDataProxyBuilder;

typedef edmplugin::PluginFactory<FW3DDataProxyBuilder*()> FW3DDataProxyBuilderFactory;

#define REGISTER_FW3DDATAPROXYBUILDER(_name_,_type_,_purpose_) \
DEFINE_PROXYBUILDER_METHODS(_name_,_type_,_purpose_); \
DEFINE_EDM_PLUGIN(FW3DDataProxyBuilderFactory,_name_,_name_::typeOfBuilder()+_name_::classTypeName()+"@"+_name_::classPurpose()+"@" #_name_)

#endif
