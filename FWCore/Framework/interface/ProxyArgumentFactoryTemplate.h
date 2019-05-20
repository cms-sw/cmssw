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
#include <cassert>
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

      ProxyArgumentFactoryTemplate(std::shared_ptr<std::pair<unsigned int, std::shared_ptr<CallbackType>>> callback)
          : callback_(std::move(callback)) {}

      ProxyArgumentFactoryTemplate(const ProxyArgumentFactoryTemplate&) = delete;
      const ProxyArgumentFactoryTemplate& operator=(const ProxyArgumentFactoryTemplate&) = delete;

      std::unique_ptr<DataProxy> makeProxy(unsigned int iovIndex) override {
        if (iovIndex != callback_->first) {
          // We are using the fact that we know all the factories related
          // to one iovIndex are dealt with before makeProxy is called with
          // a different iovIndex and that these calls are made in an
          // order such that the iovIndex increments by 1 when it changes.
          // We must clone, because we need a different callback object
          // for each iovIndex.
          assert(iovIndex == callback_->first + 1);
          callback_->first = callback_->first + 1;
          callback_->second = std::shared_ptr<CallbackType>(callback_->second->clone());
        }
        return std::make_unique<ProxyType>(callback_->second);
      }

      DataKey makeKey(const std::string& iName) const override {
        return DataKey(DataKey::makeTypeTag<typename ProxyType::ValueType>(), iName.c_str());
      }

    private:
      std::shared_ptr<std::pair<unsigned int, std::shared_ptr<CallbackType>>> callback_;
    };
  }  // namespace eventsetup
}  // namespace edm
#endif
