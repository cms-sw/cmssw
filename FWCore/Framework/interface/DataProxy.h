#ifndef FWCore_Framework_DataProxy_h
#define FWCore_Framework_DataProxy_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataProxy
//
/**\class edm::eventsetup::DataProxy

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
#include <atomic>

// user include files
#include "FWCore/Utilities/interface/thread_safety_macros.h"

// forward declarations
namespace edm {
  class ActivityRegistry;
  class EventSetupImpl;

  namespace eventsetup {
    struct ComponentDescription;
    class DataKey;
    class EventSetupRecordImpl;

    class DataProxy {
    public:
      DataProxy();
      DataProxy(DataProxy const&) = delete;
      DataProxy const& operator=(DataProxy const&) = delete;
      virtual ~DataProxy();

      // ---------- const member functions ---------------------
      bool cacheIsValid() const { return cacheIsValid_.load(std::memory_order_acquire); }

      void doGet(EventSetupRecordImpl const&,
                 DataKey const&,
                 bool iTransiently,
                 ActivityRegistry const*,
                 EventSetupImpl const*) const;
      void const* get(EventSetupRecordImpl const&,
                      DataKey const&,
                      bool iTransiently,
                      ActivityRegistry const*,
                      EventSetupImpl const*) const;

      ///returns the description of the DataProxyProvider which owns this Proxy
      ComponentDescription const* providerDescription() const { return description_; }

      // ---------- member functions ---------------------------
      void invalidate() {
        clearCacheIsValid();
        invalidateCache();
      }

      void resetIfTransient();

      void setProviderDescription(ComponentDescription const* iDesc) { description_ = iDesc; }

      virtual void initializeForNewIOV() {}

    protected:
      /**This is the function which does the real work of getting the data if it is not
          already cached.  The returning 'void const*' must point to an instance of the class
          type corresponding to the type designated in iKey. So if iKey refers to a base class interface
          the pointer must be a pointer to that base class interface and not a pointer to an inheriting class
          instance.
          */
      virtual void const* getImpl(EventSetupRecordImpl const&, DataKey const& iKey, EventSetupImpl const*) = 0;

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
      // ---------- member data --------------------------------
      ComponentDescription const* description_;
      CMS_THREAD_SAFE mutable void const* cache_;  //protected by a global mutex
      mutable std::atomic<bool> cacheIsValid_;

      // While implementing the set of code changes that enabled support
      // for concurrent IOVs, I have gone to some effort to maintain
      // the same behavior for this variable and the things that depend on
      // it. My thinking is that we are going to revisit this and make
      // changes in the not so distant future so that the transient feature
      // works again. Alternatively, we may go through and delete it and
      // everything related to it.

      // First comment is that there is only one context in which the value
      // in nonTransientAccessRequested_ is used. This is in the resetIfTransient
      // function. This function is only called immediately after invalidate
      // was called. Therefore the value is always false and condition in
      // resetIfTransient always evaluates true. So in the current code this
      // data member does nothing and has absolutely no purpose. We should
      // delete it and the associated code if we do not modify the code to
      // actually make use of the value stored sometime soon.

      // Currently, this usage occurs is when force cache clear
      // is called from EventProcessor at beginRun (which only happens
      // when a certain configuration parameter is set) and propagates down.
      // It is also used when the looper is trying to start a new loop and
      // calls resetRecordPlusDependentRecords. It is not currently used in
      // any other context.
      //
      // One other thing to note is that the virtual invalidateTransientCache
      // function is defined in this class to just call invalidateCache.
      // Outside of unit tests, the only thing that overrides this definition
      // is in CondCore/ESSources/interface/DataProxy.h. So in all other cases
      // the behavior is that invalidateCache is called twice sometimes
      // instead of just once. Possibly it is important that invalidateTransientCache
      // is called in the CondCore code. I don't know.
      mutable std::atomic<bool> nonTransientAccessRequested_;
    };
  }  // namespace eventsetup
}  // namespace edm
#endif
