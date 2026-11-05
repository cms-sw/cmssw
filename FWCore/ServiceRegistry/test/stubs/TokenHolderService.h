#ifndef FWCore_ServiceRegistry_test_stubs_TokenHolderService_h
#define FWCore_ServiceRegistry_test_stubs_TokenHolderService_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     TokenHolderService
//
/**\class TokenHolderService TokenHolderService.h FWCore/ServiceRegistry/test/stubs/TokenHolderService.h

 Description: test service that keeps a WeakServiceToken to its own service set, taken in its constructor, and
 checks that an OwningServiceToken can be taken there as well

*/

#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
}  // namespace edm

namespace testserviceregistry {
  class TokenHolderService {
  public:
    TokenHolderService(const edm::ParameterSet&, edm::ActivityRegistry&);

    edm::WeakServiceToken const& weakToken() const { return weakToken_; }
    bool ownedAtConstruction() const { return ownedAtConstruction_; }

  private:
    edm::WeakServiceToken weakToken_;
    bool ownedAtConstruction_ = false;
  };
}  // namespace testserviceregistry

#endif
