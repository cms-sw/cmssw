#ifndef PLUGINSYSTEM_REGISTRATION_MACROS_H
#define PLUGINSYSTEM_REGISTRATION_MACROS_H
// -*- C++ -*-
//
// Package:     ESSources
// Class  :     registration_macros
// 
/**\class registration_macros registration_macros.h TestCondDB/ESSources/interface/registration_macros.h

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
#include "CondCore/ESSources/interface/ProxyFactory.h"
#include "CondCore/ESSources/interface/DataProxy.h"

// forward declarations

// macros
#define INSTANTIATE_PROXY(record_, type_) template class DataProxyWrapper<record_, type_>;


#define ONLY_REGISTER_PLUGIN(record_,type_)\
typedef DataProxyWrapper<record_, type_> EDM_PLUGIN_SYM(Proxy , __LINE__ ); \
DEFINE_EDM_PLUGIN( cond::ProxyFactory, EDM_PLUGIN_SYM(Proxy , __LINE__ ), #record_ "@NewProxy")

#define REGISTER_PLUGIN(record_, type_ ) \
INSTANTIATE_PROXY(record_, type_ ) \
ONLY_REGISTER_PLUGIN(record_, type_ )

#define INSTANTIATE_PROXY_INIT(record_, type_, initializer_) template class DataProxyWrapper<record_, type_, initializer_>;

#define ONLY_REGISTER_PLUGIN_INIT(record_,type_, initializer_)\
typedef DataProxyWrapper<record_, type_, initializer_> EDM_PLUGIN_SYM(Proxy , __LINE__ ); \
DEFINE_EDM_PLUGIN( cond::ProxyFactory, EDM_PLUGIN_SYM(Proxy , __LINE__ ), #record_ "@NewProxy")

#define REGISTER_PLUGIN_INIT(record_, type_, initializer_) \
INSTANTIATE_PROXY_INIT(record_, type_, initializer_ ) \
ONLY_REGISTER_PLUGIN_INIT(record_, type_, initializer_ )



// source_ is the record name of the keyed objects
#define REGISTER_KEYLIST_PLUGIN(record_, type_, source_) \
template class DataProxyWrapper<record_, type_>; \
 struct EDM_PLUGIN_SYM(Proxy , record_ ) : public  DataProxyWrapper<record_, type_> { EDM_PLUGIN_SYM(Proxy , record_ ) () :  DataProxyWrapper<record_, type_>(#source_){};}; \
DEFINE_EDM_PLUGIN( cond::ProxyFactory, EDM_PLUGIN_SYM(Proxy , record_ ), #record_ "@NewProxy") 



#endif /* PLUGINSYSTEM_REGISTRATION_MACROS_H */
