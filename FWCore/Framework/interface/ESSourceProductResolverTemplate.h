#ifndef FWCore_Framework_ESSourceProductResolverTemplate_h
#define FWCore_Framework_ESSourceProductResolverTemplate_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ESSourceProductResolverTemplate
//
/**\class ESSourceProductResolverTemplate ESSourceProductResolverTemplate.h "FWCore/Framework/interface/ESSourceProductResolverTemplate.h"

 Description: An ESSource specific ESProductResolver which is type safe.

 Usage:
    Inherit from this class and override
      `void prefetch(edm::eventsetup::DataKey const& iKey)`
    and
       `DataT const* fetch() const`

    prefetch is guaranteed to be called before fetch where fetch should return the value obtained during the call to prefetch.
    The inheriting class must maintain the lifetime of the object returned from fetch until invalidateCache() is called.

*/
//
// Original Author:  Chris Jones
//         Created:  14/05/2020
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESSourceProductResolverNonConcurrentBase.h"

// forward declarations

namespace edm::eventsetup {
  template <typename DataT>
  class ESSourceProductResolverTemplate : public ESSourceProductResolverNonConcurrentBase {
  public:
    ESSourceProductResolverTemplate(edm::SerialTaskQueue* iQueue, std::mutex* iMutex)
        : ESSourceProductResolverNonConcurrentBase(iQueue, iMutex) {}

    ESSourceProductResolverTemplate(const ESSourceProductResolverTemplate&) = delete;
    const ESSourceProductResolverTemplate& operator=(const ESSourceProductResolverTemplate&) = delete;

    // ---------- const member functions ---------------------

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
  protected:
    /** Inheriting classes must also override 
   void prefetch(edm::eventsetup::DataKey const& iKey, EventSetupRecordDetails) override;
   */

    /** returns the data obtained in the call to prefetch */
    virtual DataT const* fetch() const = 0;

  private:
    void const* getAfterPrefetchImpl() const final { return fetch(); }
  };
}  // namespace edm::eventsetup

#endif
