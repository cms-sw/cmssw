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
  DelayedReader::getProduct(BranchKey const& k,
                            EDProductGetter const* ep,
                            ModuleCallingContext const* mcc) {

    auto sr = sharedResources_();
    std::unique_lock<SharedResourcesAcquirer> guard;
    if(sr) {
      guard =std::unique_lock<SharedResourcesAcquirer>(*sr);
    }

    if(mcc) {
      preEventReadFromSourceSignal()->emit(*(mcc->getStreamContext()),*mcc);
    }
    auto postEventReadFromSourceSig = postEventReadFromSourceSignal();
    
    auto sentryCall = [&postEventReadFromSourceSig]( ModuleCallingContext const* iContext) {
      postEventReadFromSourceSig->emit(*(iContext->getStreamContext()),*iContext);
    };
    std::unique_ptr<ModuleCallingContext const, decltype(sentryCall)> sentry(mcc, sentryCall);

    return getProduct_(k, ep);
  }

  SharedResourcesAcquirer*
  DelayedReader::sharedResources_() const {
    return nullptr;
  }
}
