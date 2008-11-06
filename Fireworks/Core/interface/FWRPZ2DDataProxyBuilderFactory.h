#ifndef Fireworks_Core_FWRPZ2DDataProxyBuilderFactory_h
#define Fireworks_Core_FWRPZ2DDataProxyBuilderFactory_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZ2DDataProxyBuilderFactory
//
/**\class FWRPZ2DDataProxyBuilderFactory FWRPZ2DDataProxyBuilderFactory.h Fireworks/Core/interface/FWRPZ2DDataProxyBuilderFactory.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Jun  5 20:13:46 EDT 2008
// $Id: FWRPZ2DDataProxyBuilderFactory.h,v 1.1 2008/06/09 19:48:44 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/register_dataproxybuilder_macro.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

// forward declarations
class FWRPZ2DDataProxyBuilder;

typedef edmplugin::PluginFactory<FWRPZ2DDataProxyBuilder*()> FWRPZ2DDataProxyBuilderFactory;

#define REGISTER_FWRPZ2DDATAPROXYBUILDER(_name_,_type_,_purpose_) \
DEFINE_PROXYBUILDER_METHODS(_name_,_type_,_purpose_); \
DEFINE_EDM_PLUGIN(FWRPZ2DDataProxyBuilderFactory,_name_,_name_::classTypeName()+"@"+_name_::classPurpose()+"@" #_name_)


#endif
