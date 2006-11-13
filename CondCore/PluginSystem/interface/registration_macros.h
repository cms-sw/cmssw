#ifndef PLUGINSYSTEM_REGISTRATION_MACROS_H
#define PLUGINSYSTEM_REGISTRATION_MACROS_H
// -*- C++ -*-
//
// Package:     PluginSystem
// Class  :     registration_macros
// 
/**\class registration_macros registration_macros.h TestCondDB/PluginSystem/interface/registration_macros.h

 Description: CPP macros used to simplify registration of plugins

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Jul 25 06:47:37 EDT 2005
// $Id: registration_macros.h,v 1.2 2006/08/01 15:08:14 xiezhen Exp $
//

// system include files

// user include files
#include "CondCore/PluginSystem/interface/ProxyFactory.h"
#include "CondCore/PluginSystem/interface/DataProxy.h"

// forward declarations

// macros
#define INSTANTIATE_PROXY(record_, type_) template class cond::DataProxy<record_, type_>;

#define ONLY_REGISTER_PLUGIN(record_, type_ ) \
typedef cond::DataProxy<record_, type_> record_ ## _ ## type_ ## _Proxy; \
DEFINE_SEAL_PLUGIN(cond::ProxyFactory, record_ ## _ ## type_ ## _Proxy, #record_ "@" #type_ "@Proxy")

#define REGISTER_PLUGIN(record_, type_ ) \
INSTANTIATE_PROXY(record_, type_ ) \
ONLY_REGISTER_PLUGIN(record_, type_ )

#endif /* PLUGINSYSTEM_REGISTRATION_MACROS_H */
