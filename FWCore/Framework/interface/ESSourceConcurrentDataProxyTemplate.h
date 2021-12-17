#ifndef FWCore_Framework_ESSourceConcurrentDataProxyTemplate_h
#define FWCore_Framework_ESSourceConcurrentDataProxyTemplate_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ESSourceConcurrentDataProxyTemplate
//
/**\class ESSourceConcurrentDataProxyTemplate ESSourceConcurrentDataProxyTemplate.h "FWCore/Framework/interface/ESSourceConcurrentDataProxyTemplate.h"

 Description: An ESSource specific DataProxy which is type safe and can run concurrently with other DataProxies from the same ESSource.

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
//         Created:  17/12/2021
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESSourceDataProxyConcurrentBase.h"

// forward declarations

namespace edm::eventsetup {
  template <typename DataT>
  class ESSourceConcurrentDataProxyTemplate : public ESSourceDataProxyConcurrentBase {
  public:
    ESSourceConcurrentDataProxyTemplate() = default;

    ESSourceConcurrentDataProxyTemplate(const ESSourceConcurrentDataProxyTemplate&) = delete;
    const ESSourceConcurrentDataProxyTemplate& operator=(const ESSourceConcurrentDataProxyTemplate&) = delete;

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
