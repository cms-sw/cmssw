#ifndef FWCore_Framework_ProxyFactoryBase_h
#define FWCore_Framework_ProxyFactoryBase_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ProxyFactoryBase
//
/**\class edm::eventsetup::ProxyFactoryBase

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

    class DataProxy;

    class ProxyFactoryBase {
    public:
      ProxyFactoryBase() = default;
      ProxyFactoryBase(const ProxyFactoryBase&) = delete;
      const ProxyFactoryBase& operator=(const ProxyFactoryBase&) = delete;
      virtual ~ProxyFactoryBase() = default;

      virtual std::unique_ptr<DataProxy> makeProxy(unsigned int iovIndex) = 0;

      virtual DataKey makeKey(const std::string& iName) const = 0;
    };
  }  // namespace eventsetup
}  // namespace edm
#endif
