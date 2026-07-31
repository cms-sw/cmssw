#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/Signal.h"
#include "FWCore/Utilities/interface/SignalSentry.h"

#include <mutex>
#include <cassert>
/*----------------------------------------------------------------------
  

----------------------------------------------------------------------*/

namespace edm {
  DelayedReader::~DelayedReader() {}

  std::shared_ptr<WrapperBase> DelayedReader::getProduct(BranchID const& k,
                                                         EDProductGetter const* ep,
                                                         ModuleCallingContext const* mcc) {
    auto preSignal = preEventReadFromSourceSignal();
    auto postSignal = postEventReadFromSourceSignal();

    auto sentry = signalslot::make_sentry([mcc, postSignal]() {
      if (mcc and postSignal) {
        postSignal->emit(*(mcc->getStreamContext()), *mcc);
      }
    });
    if (mcc and preSignal) {
      preSignal->emit(*(mcc->getStreamContext()), *mcc);
    }

    return getProduct_(k, ep);
  }

  std::pair<SharedResourcesAcquirer*, std::recursive_mutex*> DelayedReader::sharedResources_() const {
    return std::pair<SharedResourcesAcquirer*, std::recursive_mutex*>(nullptr, nullptr);
  }
}  // namespace edm
