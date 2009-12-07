#ifndef Framework_DataProxy_h
#define Framework_DataProxy_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataProxy
// 
/**\class DataProxy DataProxy.h FWCore/Framework/interface/DataProxy.h

 Description: Base class for data Proxies held by a EventSetupRecord

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 31 12:43:01 EST 2005
// $Id: DataProxy.h,v 1.7 2005/12/16 02:58:31 chrjones Exp $
//

// system include files

// user include files

// forward declarations
namespace edm {
   namespace eventsetup {
      class EventSetupRecord;
      class DataKey;
      class ComponentDescription;
      
class DataProxy
{

   public:
      DataProxy();
      virtual ~DataProxy();

      // ---------- const member functions ---------------------
      bool cacheIsValid() const { return cacheIsValid_; }
      
      virtual void doGet(const EventSetupRecord& iRecord, const DataKey& iKey) const = 0;

      ///returns the description of the DataProxyProvider which owns this Proxy
      const ComponentDescription* providerDescription() const {
         return description_;
      }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void invalidate() {
         clearCacheIsValid();
         invalidateCache();
      }
      
      void resetIfTransient();

      void setProviderDescription(const ComponentDescription* iDesc) {
         description_ = iDesc;
      }
   protected:
      /** indicates that the Proxy should invalidate any cached information
           as that information has 'expired'
         */
      virtual void invalidateCache() = 0;

      void setCacheIsValidAndAccessType(bool iTransientAccessOnly);
      void clearCacheIsValid();      
   private:
      DataProxy(const DataProxy&); // stop default

      const DataProxy& operator=(const DataProxy&); // stop default

      // ---------- member data --------------------------------
      bool cacheIsValid_;
      bool nonTransientAccessRequested_;
      const ComponentDescription* description_;
};
   }
}
#endif
