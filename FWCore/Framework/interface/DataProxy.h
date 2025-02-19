#ifndef FWCore_Framework_DataProxy_h
#define FWCore_Framework_DataProxy_h
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
      struct ComponentDescription;
      class DataKey;
      class EventSetupRecord;

      class DataProxy {

      public:
         DataProxy();
         virtual ~DataProxy();

         // ---------- const member functions ---------------------
         bool cacheIsValid() const { return cacheIsValid_; }

         void doGet(EventSetupRecord const& iRecord, DataKey const& iKey, bool iTransiently) const;
         void const* get(EventSetupRecord const&, DataKey const& iKey, bool iTransiently) const;

         ///returns the description of the DataProxyProvider which owns this Proxy
         ComponentDescription const* providerDescription() const {
            return description_;
         }
         // ---------- static member functions --------------------

         // ---------- member functions ---------------------------
         void invalidate() {
            clearCacheIsValid();
            invalidateCache();
         }

         void resetIfTransient();

         void setProviderDescription(ComponentDescription const* iDesc) {
            description_ = iDesc;
         }
      protected:
         /**This is the function which does the real work of getting the data if it is not
          already cached.  The returning 'void const*' must point to an instance of the class
          type corresponding to the type designated in iKey. So if iKey refers to a base class interface
          the pointer must be a pointer to that base class interface and not a pointer to an inheriting class
          instance.
          */
         virtual void const* getImpl(EventSetupRecord const&, DataKey const& iKey) =0;

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
         DataProxy(DataProxy const&); // stop default

         DataProxy const& operator=(DataProxy const&); // stop default
         void setCacheIsValidAndAccessType(bool iTransientAccessOnly) const;

         // ---------- member data --------------------------------
         mutable void const* cache_;
         mutable bool cacheIsValid_;
         mutable bool nonTransientAccessRequested_;
         ComponentDescription const* description_;
      };
   }
}
#endif
