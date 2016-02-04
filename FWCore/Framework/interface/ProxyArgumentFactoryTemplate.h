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
class ProxyArgumentFactoryTemplate : public ProxyFactoryBase
{

   public:
      typedef typename T::record_type record_type;

      ProxyArgumentFactoryTemplate(ArgT iArg) : arg_(iArg) {}
      //virtual ~ProxyArgumentFactoryTemplate()

      // ---------- const member functions ---------------------
      virtual std::auto_ptr<DataProxy> makeProxy() const {
         return std::auto_ptr<DataProxy>(new T(arg_));
      }
            
      virtual DataKey makeKey(const std::string& iName) const {
         return DataKey(DataKey::makeTypeTag< typename T::value_type>(),iName.c_str());
      }
      
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      ProxyArgumentFactoryTemplate(const ProxyArgumentFactoryTemplate&); // stop default

      const ProxyArgumentFactoryTemplate& operator=(const ProxyArgumentFactoryTemplate&); // stop default

      // ---------- member data --------------------------------
      mutable ArgT arg_;
};

   }
}
#endif
