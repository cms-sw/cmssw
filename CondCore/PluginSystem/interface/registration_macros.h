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
//

// system include files

// user include files
#include "CondCore/PluginSystem/interface/ProxyFactory.h"
#include "CondCore/PluginSystem/interface/OldDataProxy.h"
#include "CondCore/PluginSystem/interface/DataProxy.h"

// forward declarations

// macros
#define INSTANTIATE_PROXY(record_, type_) template class OldDataProxy<record_, type_>;template class DataProxy<record_, type_>;

#define ONLY_REGISTER_PLUGIN(record_,type_)\
typedef OldDataProxy<record_, type_> EDM_PLUGIN_SYM(OldProxy , __LINE__ ); \
typedef DataProxy<record_, type_> EDM_PLUGIN_SYM(Proxy , __LINE__ ); \
DEFINE_EDM_PLUGIN( oldcond::ProxyFactory, EDM_PLUGIN_SYM(OldProxy , __LINE__ ), #record_ "@" #type_ "@Proxy")

// DEFINE_EDM_PLUGIN( cond::ProxyFactory, EDM_PLUGIN_SYM(Proxy , __LINE__ ), #record_ "@NewProxy")

#define REGISTER_PLUGIN(record_, type_ ) \
INSTANTIATE_PROXY(record_, type_ ) \
ONLY_REGISTER_PLUGIN(record_, type_ )

#endif /* PLUGINSYSTEM_REGISTRATION_MACROS_H */
