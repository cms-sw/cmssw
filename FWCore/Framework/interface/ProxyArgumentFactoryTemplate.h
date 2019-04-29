#ifndef FWCore_Framework_ProxyArgumentFactoryTemplate_h
#define FWCore_Framework_ProxyArgumentFactoryTemplate_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ProxyArgumentFactoryTemplate
//
/**\class edm::eventsetup::ProxyArgumentFactoryTemplate

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Mon Apr 11 16:20:52 CDT 2005
//

// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/ProxyFactoryBase.h"
#include "FWCore/Framework/interface/DataKey.h"

namespace edm {
  namespace eventsetup {

    class DataProxy;

    template <class ProxyType, class CallbackType>
    class ProxyArgumentFactoryTemplate : public ProxyFactoryBase {

    public:

      using RecordType = typename ProxyType::RecordType;

      ProxyArgumentFactoryTemplate(std::shared_ptr<std::vector<std::shared_ptr<CallbackType>>> callbacks) : callbacks_(std::move(callbacks)) {}

      ProxyArgumentFactoryTemplate(const ProxyArgumentFactoryTemplate&) = delete;
      const ProxyArgumentFactoryTemplate& operator=(const ProxyArgumentFactoryTemplate&) = delete;

      std::unique_ptr<DataProxy> makeProxy(unsigned int iovIndex) override {
        while (iovIndex >= callbacks_->size()) {
          callbacks_->push_back(std::make_shared<CallbackType>(callbacks_->at(0)->get(),
                                                               callbacks_->at(0)->method(),
                                                               callbacks_->at(0)->transitionID(),
                                                               callbacks_->at(0)->decorator()));
        }
        return std::make_unique<ProxyType>(callbacks_->at(iovIndex));
      }

      DataKey makeKey(const std::string& iName) const override {
        return DataKey(DataKey::makeTypeTag<typename ProxyType::ValueType>(), iName.c_str());
      }

    private:

      std::shared_ptr<std::vector<std::shared_ptr<CallbackType>>> callbacks_;
    };
  }
}
#endif
