#ifndef EVENTSETUPPRODUCER_PROXYFACTORYBASE_H
#define EVENTSETUPPRODUCER_PROXYFACTORYBASE_H
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ProxyFactoryBase
// 
/**\class ProxyFactoryBase ProxyFactoryBase.h FWCore/Framework/interface/ProxyFactoryBase.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Apr  7 21:50:36 CDT 2005
// $Id: ProxyFactoryBase.h,v 1.2 2005/06/23 19:59:30 wmtan Exp $
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/DataKey.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      class DataProxy;
class ProxyFactoryBase
{

   public:
      ProxyFactoryBase() {}
      virtual ~ProxyFactoryBase() {}

      // ---------- const member functions ---------------------
      virtual std::auto_ptr<DataProxy> makeProxy() const = 0;
      
      virtual DataKey makeKey(const std::string& iName) const = 0;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      ProxyFactoryBase(const ProxyFactoryBase&); // stop default

      const ProxyFactoryBase& operator=(const ProxyFactoryBase&); // stop default

      // ---------- member data --------------------------------

};

   }
}

#endif /* EVENTSETUPPRODUCER_PROXYFACTORYBASE_H */
