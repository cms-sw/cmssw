#include "FWCore/ServiceRegistry/test/stubs/TokenHolderService.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

namespace testserviceregistry {
  TokenHolderService::TokenHolderService(const edm::ParameterSet&, edm::ActivityRegistry&)
      : weakToken_(edm::ServiceRegistry::instance().presentToken()) {
    // sharing the ownership of the service set being constructed must work, but a service must not keep an
    // OwningServiceToken to its own set, as it would never be destroyed
    edm::OwningServiceToken owner(edm::ServiceRegistry::instance().presentToken());
    ownedAtConstruction_ = bool(owner) and bool(weakToken_.lock());
  }
}  // namespace testserviceregistry
