#include "FWCore/Framework/interface/ComponentConstructionSentry.h"
#include "FWCore/Framework/interface/ComponentInterfaceHolder.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm {
  namespace eventsetup {

    ComponentConstructionSentry::ComponentConstructionSentry(ComponentInterfaceHolder const& iInterfaceHolder,
                                                             ComponentDescription const& iDescription)
        : interfaceHolder_(iInterfaceHolder), description_(iDescription) {
      interfaceHolder_.preConstructionSignal().emit(description_);
      // Here would be where the construction signal would go
    }

    ComponentConstructionSentry::~ComponentConstructionSentry() noexcept(false) {
      try {
        interfaceHolder_.postConstructionSignal().emit(description_);
      } catch (...) {
        if (succeeded_) {
          // no exception happened during construction so can rethrow the exception from the post construction signal
          throw;
        }
      }
    }

  }  // namespace eventsetup
}  // namespace edm
