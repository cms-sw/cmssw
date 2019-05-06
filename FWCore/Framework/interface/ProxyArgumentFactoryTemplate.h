#ifndef Framework_ProxyArgumentFactoryTemplate_h
#define Framework_ProxyArgumentFactoryTemplate_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ProxyArgumentFactoryTemplate
//
/**\class ProxyArgumentFactoryTemplate ProxyArgumentFactoryTemplate.h FWCore/Framework/interface/ProxyArgumentFactoryTemplate.h

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

// user include files
#include "FWCore/Framework/interface/ProxyFactoryBase.h"
#include "FWCore/Framework/interface/DataKey.h"

// forward declarations
namespace edm {
  namespace eventsetup {

    template <class T, class ArgT>
    class ProxyArgumentFactoryTemplate : public ProxyFactoryBase {
    public:
      typedef typename T::record_type record_type;

      ProxyArgumentFactoryTemplate(ArgT iArg) : arg_(iArg) {}
      //virtual ~ProxyArgumentFactoryTemplate()

      // ---------- const member functions ---------------------
      std::unique_ptr<DataProxy> makeProxy() const override { return std::make_unique<T>(arg_); }

      DataKey makeKey(const std::string& iName) const override {
        return DataKey(DataKey::makeTypeTag<typename T::value_type>(), iName.c_str());
      }

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

    private:
      ProxyArgumentFactoryTemplate(const ProxyArgumentFactoryTemplate&) = delete;  // stop default

      const ProxyArgumentFactoryTemplate& operator=(const ProxyArgumentFactoryTemplate&) = delete;  // stop default

      // ---------- member data --------------------------------
      mutable ArgT arg_;
    };

  }  // namespace eventsetup
}  // namespace edm
#endif
