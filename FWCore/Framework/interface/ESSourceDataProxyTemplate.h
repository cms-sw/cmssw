#ifndef FWCore_Framework_ESSourceDataProxyTemplate_h
#define FWCore_Framework_ESSourceDataProxyTemplate_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ESSourceDataProxyTemplate
//
/**\class ESSourceDataProxyTemplate ESSourceDataProxyTemplate.h "FWCore/Framework/interface/ESSourceDataProxyTemplate.h"

 Description: An ESSource specific DataProxy which is type safe.

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
#include "FWCore/Framework/interface/ESSourceDataProxyBase.h"

// forward declarations

namespace edm::eventsetup {
  template <typename DataT>
  class ESSourceDataProxyTemplate : public ESSourceDataProxyBase {
  public:
    ESSourceDataProxyTemplate(edm::SerialTaskQueue* iQueue, std::mutex* iMutex)
        : ESSourceDataProxyBase(iQueue, iMutex) {}

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

    ESSourceDataProxyTemplate(const ESSourceDataProxyTemplate&) = delete;

    const ESSourceDataProxyTemplate& operator=(const ESSourceDataProxyTemplate&) = delete;
  };
}  // namespace edm::eventsetup

#endif
