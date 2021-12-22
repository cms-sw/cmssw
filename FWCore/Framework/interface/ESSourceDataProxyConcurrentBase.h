#ifndef FWCore_Framework_ESSourceDataProxyConcurrentBase_h
#define FWCore_Framework_ESSourceDataProxyConcurrentBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ESSourceDataProxyConcurrentBase
//
/**\class ESSourceDataProxyConcurrentBase ESSourceDataProxyConcurrentBase.h "FWCore/Framework/interface/ESSourceDataProxyConcurrentBase.h"

 Description: Base class for DataProxies for ESSources that require no synchronization

 Usage:
    The ESSourceDataProxyConcurrentBase allows DataProxies from the same ESSource to be called concurrently.

    NOTE: if inheriting classes override `void invalidateCache()` they must be sure to call this classes
    implementation as part of the call.

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
  class ESSourceDataProxyConcurrentBase : public ESSourceDataProxyBase {
  public:
    ESSourceDataProxyConcurrentBase() {}

  private:
    void prefetchAsyncImpl(edm::WaitingTaskHolder iTask,
                           edm::eventsetup::EventSetupRecordImpl const& iES,
                           edm::eventsetup::DataKey const& iKey,
                           edm::EventSetupImpl const*,
                           edm::ServiceToken const&,
                           edm::ESParentContext const&) final;

    // ---------- member data --------------------------------
  };
}  // namespace edm::eventsetup
#endif
