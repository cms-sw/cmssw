#ifndef FWCore_Framework_ESProductResolverFactoryBase_h
#define FWCore_Framework_ESProductResolverFactoryBase_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProductResolverFactoryBase
//
/**\class edm::eventsetup::ESProductResolverFactoryBase

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Apr  7 21:50:36 CDT 2005
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/DataKey.h"

namespace edm {
  namespace eventsetup {

    class ESProductResolver;

    class ESProductResolverFactoryBase {
    public:
      ESProductResolverFactoryBase() = default;
      ESProductResolverFactoryBase(const ESProductResolverFactoryBase&) = delete;
      const ESProductResolverFactoryBase& operator=(const ESProductResolverFactoryBase&) = delete;
      virtual ~ESProductResolverFactoryBase() = default;

      virtual std::unique_ptr<ESProductResolver> makeResolver(unsigned int iovIndex) = 0;

      virtual DataKey makeKey(const std::string& iName) const = 0;
    };
  }  // namespace eventsetup
}  // namespace edm
#endif
