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
    This class defines the interface used to handle retrieving data from an
 EventSetup Record. 

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 31 12:43:01 EST 2005
//

// system include files

// user include files

// forward declarations
namespace edm {
   namespace eventsetup {
      class EventSetupRecord;
      class DataKey;
      class ComponentDescription;
      
      class DataProxy {
         
      public:
         DataProxy();
         virtual ~DataProxy();
         
         // ---------- const member functions ---------------------
         bool cacheIsValid() const { return cacheIsValid_; }
         
         void doGet(const EventSetupRecord& iRecord, const DataKey& iKey, bool iTransiently) const;
         const void* get(const EventSetupRecord&, const DataKey& iKey, bool iTransiently) const;
         
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
         /**This is the function which does the real work of getting the data if it is not
          already cached.  The returning 'const void*' must point to an instance of the class
          type corresponding to the type designated in iKey. So if iKey refers to a base class interface
          the pointer must be a pointer to that base class interface and not a pointer to an inheriting class
          instance.
          */
         virtual const void* getImpl(const EventSetupRecord&, const DataKey& iKey) =0;
         
         /** indicates that the Proxy should invalidate any cached information
          as that information has 'expired' (i.e. we have moved to a new IOV)
          */
         virtual void invalidateCache() = 0;
         
         /** indicates that the Proxy should invalidate any cached information
          as that information was accessed transiently and therefore is not
          intended to be kept over the entire IOV.  Default is to call
          invalidateCache().
          */
         virtual void invalidateTransientCache();
         
         void clearCacheIsValid();      
      private:
         DataProxy(const DataProxy&); // stop default
         
         const DataProxy& operator=(const DataProxy&); // stop default
         void setCacheIsValidAndAccessType(bool iTransientAccessOnly) const;
         
         // ---------- member data --------------------------------
         mutable const void* cache_;
         mutable bool cacheIsValid_;
         mutable bool nonTransientAccessRequested_;
         const ComponentDescription* description_;
      };
   }
}
#endif
