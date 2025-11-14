#include "FWCore/Framework/interface/ComponentConstructionSentry.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm {
  namespace eventsetup {

    ComponentConstructionSentry::ComponentConstructionSentry(EventSetupProvider const& iProvider,
                                                             ComponentDescription const& iDescription)
        : provider_(iProvider), description_(iDescription) {
      provider_.activityRegistry()->preESModuleConstructionSignal_(description_);
      // Here would be where the construction signal would go
    }

    ComponentConstructionSentry::~ComponentConstructionSentry() noexcept(false) {
      try {
        provider_.activityRegistry()->postESModuleConstructionSignal_(description_);
      } catch (...) {
        if (succeeded_) {
          // no exception happened during construction so can rethrow the exception from the post construction signal
          throw;
        }
      }
    }

  }  // namespace eventsetup
}  // namespace edm