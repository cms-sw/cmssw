#ifndef FWCore_Framework_ESProductResolverFactoryTemplate_h
#define FWCore_Framework_ESProductResolverFactoryTemplate_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProductResolverFactoryTemplate
//
/**\class edm::eventsetup::ESProductResolverFactoryTemplate

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
#include "FWCore/Framework/interface/ESProductResolverFactoryBase.h"
#include "FWCore/Framework/interface/DataKey.h"

namespace edm {
  namespace eventsetup {

    class ESProductResolver;

    template <class T>
    class ESProductResolverFactoryTemplate : public ESProductResolverFactoryBase {
    public:
      using RecordType = typename T::record_type;

      ESProductResolverFactoryTemplate() = default;

      ESProductResolverFactoryTemplate(const ESProductResolverFactoryTemplate&) = delete;
      const ESProductResolverFactoryTemplate& operator=(const ESProductResolverFactoryTemplate&) = delete;

      std::unique_ptr<ESProductResolver> makeResolver(unsigned int) override { return std::make_unique<T>(); }

      DataKey makeKey(const std::string& iName) const override {
        return DataKey(DataKey::makeTypeTag<typename T::value_type>(), iName.c_str());
      }
    };
  }  // namespace eventsetup
}  // namespace edm
#endif
