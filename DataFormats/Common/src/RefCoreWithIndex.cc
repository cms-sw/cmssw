#include "DataFormats/Common/interface/RefCoreWithIndex.h"
#include "DataFormats/Common/interface/RefCore.h"

namespace edm {

  RefCoreWithIndex::RefCoreWithIndex(ProductID const& theId,
                                     void const* prodPtr,
                                     EDProductGetter const* prodGetter,
                                     bool transient,
                                     unsigned int iIndex)
      : cachePtr_(prodPtr),
        processIndex_(theId.processIndex()),
        productIndex_(theId.productIndex()),
        elementIndex_(iIndex) {
    if (transient) {
      setTransient();
    }
    if (prodPtr == nullptr && prodGetter != nullptr) {
      setCacheIsProductGetter(prodGetter);
    }
  }

  RefCoreWithIndex::RefCoreWithIndex(RefCore const& iCore, unsigned int iIndex)
      : processIndex_(iCore.processIndex_), productIndex_(iCore.productIndex_), elementIndex_(iIndex) {
    cachePtr_.store(iCore.cachePtr_.load(std::memory_order_relaxed), std::memory_order_relaxed);
  }

  RefCoreWithIndex::RefCoreWithIndex(RefCoreWithIndex const& iOther)
      : processIndex_(iOther.processIndex_), productIndex_(iOther.productIndex_), elementIndex_(iOther.elementIndex_) {
    cachePtr_.store(iOther.cachePtr_.load(std::memory_order_relaxed), std::memory_order_relaxed);
  }

  RefCoreWithIndex& RefCoreWithIndex::operator=(RefCoreWithIndex const& iOther) {
    cachePtr_.store(iOther.cachePtr_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    processIndex_ = iOther.processIndex_;
    productIndex_ = iOther.productIndex_;
    elementIndex_ = iOther.elementIndex_;
    return *this;
  }

}  // namespace edm
