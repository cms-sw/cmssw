#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"

#include <mutex>
#include <cassert>
/*----------------------------------------------------------------------
  

----------------------------------------------------------------------*/


namespace edm {
  DelayedReader::~DelayedReader() {}

  std::unique_ptr<EDProduct> 
  DelayedReader::getProduct(BranchKey const& k, EDProductGetter const* ep) {
    auto sr = sharedResources_();
    std::unique_lock<SharedResourcesAcquirer> guard;
    if(sr) {
      guard =std::unique_lock<SharedResourcesAcquirer>(*sr);
    }
    return getProduct_(k, ep);
  }

  SharedResourcesAcquirer*
  DelayedReader::sharedResources_() const {
    return nullptr;
  }
}
