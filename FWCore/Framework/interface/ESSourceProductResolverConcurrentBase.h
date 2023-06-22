#ifndef FWCore_Framework_ESSourceProductResolverConcurrentBase_h
#define FWCore_Framework_ESSourceProductResolverConcurrentBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     ESSourceProductResolverConcurrentBase
//
/**\class ESSourceProductResolverConcurrentBase ESSourceProductResolverConcurrentBase.h "FWCore/Framework/interface/ESSourceProductResolverConcurrentBase.h"

 Description: Base class for DataProxies for ESSources that require no synchronization

 Usage:
    The ESSourceProductResolverConcurrentBase allows DataProxies from the same ESSource to be called concurrently.

    NOTE: if inheriting classes override `void invalidateCache()` they must be sure to call this classes
    implementation as part of the call.

*/
//
// Original Author:  Chris Jones
//         Created:  14/05/2020
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESSourceProductResolverBase.h"

// forward declarations

namespace edm::eventsetup {
  class ESSourceProductResolverConcurrentBase : public ESSourceProductResolverBase {
  public:
    ESSourceProductResolverConcurrentBase() {}

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
