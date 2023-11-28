#ifndef FWCore_Framework_ESProductResolverArgumentFactoryTemplate_h
#define FWCore_Framework_ESProductResolverArgumentFactoryTemplate_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProductResolverArgumentFactoryTemplate
//
/**\class edm::eventsetup::ESProductResolverArgumentFactoryTemplate

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
#include "FWCore/Framework/interface/ESProductResolverFactoryBase.h"
#include "FWCore/Framework/interface/DataKey.h"

namespace edm {
  namespace eventsetup {

    class ESProductResolver;

    template <class ResolverType, class CallbackType>
    class ESProductResolverArgumentFactoryTemplate : public ESProductResolverFactoryBase {
    public:
      using RecordType = typename ResolverType::RecordType;

      ESProductResolverArgumentFactoryTemplate(
          std::shared_ptr<std::pair<unsigned int, std::shared_ptr<CallbackType>>> callback)
          : callback_(std::move(callback)) {}

      ESProductResolverArgumentFactoryTemplate(const ESProductResolverArgumentFactoryTemplate&) = delete;
      const ESProductResolverArgumentFactoryTemplate& operator=(const ESProductResolverArgumentFactoryTemplate&) =
          delete;

      std::unique_ptr<ESProductResolver> makeResolver(unsigned int iovIndex) override {
        if (iovIndex != callback_->first) {
          // We are using the fact that we know all the factories related
          // to one iovIndex are dealt with before makeResolver is called with
          // a different iovIndex and that these calls are made in an
          // order such that the iovIndex increments by 1 when it changes.
          // We must clone, because we need a different callback object
          // for each iovIndex.
          assert(iovIndex == callback_->first + 1);
          callback_->first = callback_->first + 1;
          callback_->second = std::shared_ptr<CallbackType>(callback_->second->clone());
        }
        return std::make_unique<ResolverType>(callback_->second);
      }

      DataKey makeKey(const std::string& iName) const override {
        return DataKey(DataKey::makeTypeTag<typename ResolverType::ValueType>(), iName.c_str());
      }

    private:
      std::shared_ptr<std::pair<unsigned int, std::shared_ptr<CallbackType>>> callback_;
    };
  }  // namespace eventsetup
}  // namespace edm
#endif
