#ifndef Fireworks_Core_FW3DLegoDataProxyBuilderFactory_h
#define Fireworks_Core_FW3DLegoDataProxyBuilderFactory_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DLegoDataProxyBuilderFactory
//
/**\class FW3DLegoDataProxyBuilderFactory FW3DLegoDataProxyBuilderFactory.h Fireworks/Core/interface/FW3DLegoDataProxyBuilderFactory.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Jun  5 20:13:55 EDT 2008
// $Id: FW3DLegoDataProxyBuilderFactory.h,v 1.1 2008/06/09 19:48:45 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/register_dataproxybuilder_macro.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

// forward declarations

class FW3DLegoDataProxyBuilder;

typedef edmplugin::PluginFactory<FW3DLegoDataProxyBuilder*()> FW3DLegoDataProxyBuilderFactory;

#define REGISTER_FW3DLEGODATAPROXYBUILDER(_name_,_type_,_purpose_) \
DEFINE_PROXYBUILDER_METHODS(_name_,_type_,_purpose_); \
DEFINE_EDM_PLUGIN(FW3DLegoDataProxyBuilderFactory,_name_,_name_::classTypeName()+"@"+_name_::classPurpose()+"@" #_name_)

#endif
