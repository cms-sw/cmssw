#ifndef CONDCORE_ESSOURCES_REGISTRATION_MACROS_H
#define CONDCORE_ESSOURCES_REGISTRATION_MACROS_H
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
#include "CondCore/ESSources/interface/ProductResolverFactory.h"
#include "CondCore/ESSources/interface/ProductResolver.h"
#include "CondFormats/SerializationHelper/interface/SerializationHelperFactory.h"

// forward declarations

// macros
#define INSTANTIATE_RESOLVER(record_, type_) template class ProductResolverWrapper<record_, type_>;

#define ONLY_REGISTER_PLUGIN(record_, type_)                                                \
  typedef ProductResolverWrapper<record_, type_> EDM_PLUGIN_SYM(ProductResolver, __LINE__); \
  DEFINE_EDM_PLUGIN2(cond::ProductResolverFactory, EDM_PLUGIN_SYM(ProductResolver, __LINE__), #record_ "@NewProxy")

#define REGISTER_PLUGIN(record_, type_)      \
  INSTANTIATE_RESOLVER(record_, type_)       \
  DEFINE_COND_SERIAL_REGISTER_PLUGIN(type_); \
  ONLY_REGISTER_PLUGIN(record_, type_)

#define REGISTER_PLUGIN_NO_SERIAL(record_, type_) \
  INSTANTIATE_RESOLVER(record_, type_)            \
  ONLY_REGISTER_PLUGIN(record_, type_)

#define INSTANTIATE_RESOLVER_INIT(record_, type_, initializer_) \
  template class ProductResolverWrapper<record_, type_, initializer_>;

#define ONLY_REGISTER_PLUGIN_INIT(record_, type_, initializer_)                                           \
  typedef ProductResolverWrapper<record_, type_, initializer_> EDM_PLUGIN_SYM(ProductResolver, __LINE__); \
  DEFINE_EDM_PLUGIN2(cond::ProductResolverFactory, EDM_PLUGIN_SYM(ProductResolver, __LINE__), #record_ "@NewProxy")

#define REGISTER_PLUGIN_INIT(record_, type_, initializer_)      \
  INSTANTIATE_RESOLVER_INIT(record_, type_, initializer_)       \
  DEFINE_COND_SERIAL_REGISTER_PLUGIN_INIT(type_, initializer_); \
  ONLY_REGISTER_PLUGIN_INIT(record_, type_, initializer_)

#define REGISTER_PLUGIN_NO_SERIAL_INIT(record_, type_, initializer_) \
  INSTANTIATE_RESOLVER_INIT(record_, type_, initializer_)            \
  ONLY_REGISTER_PLUGIN_INIT(record_, type_, initializer_)

// source_ is the record name of the keyed objects
#define REGISTER_KEYLIST_PLUGIN(record_, type_, source_)                                             \
  template class ProductResolverWrapper<record_, type_>;                                             \
  struct EDM_PLUGIN_SYM(ProductResolver, record_) : public ProductResolverWrapper<record_, type_> {  \
    EDM_PLUGIN_SYM(ProductResolver, record_)() : ProductResolverWrapper<record_, type_>(#source_){}; \
  };                                                                                                 \
  DEFINE_EDM_PLUGIN(cond::ProductResolverFactory, EDM_PLUGIN_SYM(ProductResolver, record_), #record_ "@NewProxy")

#endif /* CONDCORE_ESSOURCES_REGISTRATION_MACROS_H */
