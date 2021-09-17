#ifndef FWCore_Framework_ProxyFactoryTemplate_h
#define FWCore_Framework_ProxyFactoryTemplate_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ProxyFactoryTemplate
//
/**\class edm::eventsetup::ProxyFactoryTemplate

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Apr  8 07:59:32 CDT 2005
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/ProxyFactoryBase.h"
#include "FWCore/Framework/interface/DataKey.h"

namespace edm {
  namespace eventsetup {

    class DataProxy;

    template <class T>
    class ProxyFactoryTemplate : public ProxyFactoryBase {
    public:
      using RecordType = typename T::record_type;

      ProxyFactoryTemplate() = default;

      ProxyFactoryTemplate(const ProxyFactoryTemplate&) = delete;
      const ProxyFactoryTemplate& operator=(const ProxyFactoryTemplate&) = delete;

      std::unique_ptr<DataProxy> makeProxy(unsigned int) override { return std::make_unique<T>(); }

      DataKey makeKey(const std::string& iName) const override {
        return DataKey(DataKey::makeTypeTag<typename T::value_type>(), iName.c_str());
      }
    };
  }  // namespace eventsetup
}  // namespace edm
#endif
