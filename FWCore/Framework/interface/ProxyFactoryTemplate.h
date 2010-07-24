#ifndef Framework_ProxyFactoryTemplate_h
#define Framework_ProxyFactoryTemplate_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ProxyFactoryTemplate
// 
/**\class ProxyFactoryTemplate ProxyFactoryTemplate.h FWCore/Framework/interface/ProxyFactoryTemplate.h

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

// forward declarations
namespace edm {
   namespace eventsetup {

template <class T>
class ProxyFactoryTemplate : public ProxyFactoryBase
{

   public:
      typedef typename T::record_type record_type;
   
      ProxyFactoryTemplate() {}
      //virtual ~ProxyFactoryTemplate();

      // ---------- const member functions ---------------------
      virtual std::auto_ptr<DataProxy> makeProxy() const {
         return std::auto_ptr<DataProxy>(new T);
      }
      
      
      virtual DataKey makeKey(const std::string& iName) const {
         return DataKey(DataKey::makeTypeTag< typename T::value_type>(),iName.c_str());
      }
      
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      ProxyFactoryTemplate(const ProxyFactoryTemplate&); // stop default

      const ProxyFactoryTemplate& operator=(const ProxyFactoryTemplate&); // stop default

      // ---------- member data --------------------------------

};

   }
}

#endif
