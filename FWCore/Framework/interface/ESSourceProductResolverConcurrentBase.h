// -*- C++ -*-
#ifndef FWCore_Framework_ESSourceProductResolverConcurrentBase_h
#define FWCore_Framework_ESSourceProductResolverConcurrentBase_h
//
// Package:     FWCore/Framework
// Class  :     ESSourceProductResolverConcurrentBase
//
/**\class edm::eventsetup::ESSourceProductResolverConcurrentBase

 Description: Base class for ESProductResolver for ESSources that require no synchronization

 Usage:
    The ESSourceProductResolverConcurrentBase allows ESProductResolvers from the same ESSource to be called concurrently.

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
                           edm::ESParentContext const&) noexcept final;

    // ---------- member data --------------------------------
  };
}  // namespace edm::eventsetup
#endif
