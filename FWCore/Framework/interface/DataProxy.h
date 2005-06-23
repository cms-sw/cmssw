#ifndef EVENTSETUP_DATAPROXY_H
#define EVENTSETUP_DATAPROXY_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     DataProxy
// 
/**\class DataProxy DataProxy.h Core/CoreFramework/interface/DataProxy.h

 Description: Base class for data Proxies held by a EventSetupRecord

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 31 12:43:01 EST 2005
// $Id: DataProxy.h,v 1.1 2005/05/29 02:29:53 wmtan Exp $
//

// system include files

// user include files

// forward declarations
namespace edm {
   namespace eventsetup {
      
class DataProxy
{

   public:
      DataProxy();
      virtual ~DataProxy();

      // ---------- const member functions ---------------------
      bool cacheIsValid() const { return cacheIsValid_; }
      
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void invalidate() {
         clearCacheIsValid();
         invalidateCache();
      }

   protected:
      /** indicates that the Proxy should invalidate any cached information
           as that information has 'expired'
         */
      virtual void invalidateCache() = 0;

      void setCacheIsValid() { cacheIsValid_ = true; }
      void clearCacheIsValid() { cacheIsValid_ = false; }
      
   private:
      DataProxy(const DataProxy&); // stop default

      const DataProxy& operator=(const DataProxy&); // stop default

      // ---------- member data --------------------------------
      bool cacheIsValid_;
};
   }
}
#endif /* EVENTSETUP_DATAPROXY_H */
