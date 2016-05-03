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
  DelayedReader::getProduct(BranchKey const& k, EDProductGetter const* ep) {
    auto sr = sharedResources_();
    std::unique_lock<SharedResourcesAcquirer> guard;
    if(sr) {
      guard =std::unique_lock<SharedResourcesAcquirer>(*sr);
    }
    return getProduct_(k, ep);
  }

  std::unique_ptr<WrapperBase>
  DelayedReader::getProduct(BranchKey const& k,
                            EDProductGetter const* ep,
                            signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const& preReadFromSourceSignal,
                            signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const& postReadFromSourceSignal,
                            ModuleCallingContext const* mcc) {

    auto sr = sharedResources_();
    std::unique_lock<SharedResourcesAcquirer> guard;
    if(sr) {
      guard =std::unique_lock<SharedResourcesAcquirer>(*sr);
    }

    if(mcc) {
      preReadFromSourceSignal.emit(*(mcc->getStreamContext()),*mcc);
    }
    std::shared_ptr<void> guardForSignal(nullptr,[&postReadFromSourceSignal,mcc](const void*){
      if(mcc) {
        postReadFromSourceSignal.emit(*(mcc->getStreamContext()),*mcc);
      }
    });

    return getProduct_(k, ep);
  }

  SharedResourcesAcquirer*
  DelayedReader::sharedResources_() const {
    return nullptr;
  }
}
