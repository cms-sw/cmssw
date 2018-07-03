#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/Signal.h"

#include <mutex>
#include <cassert>
/*----------------------------------------------------------------------
  

----------------------------------------------------------------------*/


namespace edm {
  DelayedReader::~DelayedReader() {}

  std::unique_ptr<WrapperBase>
  DelayedReader::getProduct(BranchID const& k,
                            EDProductGetter const* ep,
                            ModuleCallingContext const* mcc) {

    auto preSignal = preEventReadFromSourceSignal();
    if(mcc and preSignal) {
      preSignal->emit(*(mcc->getStreamContext()),*mcc);
    }
    auto postSignal = postEventReadFromSourceSignal();
    
    auto sentryCall = [&postSignal]( ModuleCallingContext const* iContext) {
      if(postSignal) {
        postSignal->emit(*(iContext->getStreamContext()),*iContext);
      }
    };
    std::unique_ptr<ModuleCallingContext const, decltype(sentryCall)> sentry(mcc, sentryCall);

    return getProduct_(k, ep);
  }

  std::pair<SharedResourcesAcquirer*, std::recursive_mutex*>
  DelayedReader::sharedResources_() const {
    return std::pair<SharedResourcesAcquirer*, std::recursive_mutex*>(nullptr, nullptr);
  }
}
