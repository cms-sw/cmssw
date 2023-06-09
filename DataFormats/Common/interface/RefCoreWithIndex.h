#ifndef DataFormats_Common_RefCoreWithIndex_h
#define DataFormats_Common_RefCoreWithIndex_h

/*----------------------------------------------------------------------
  
RefCoreWithIndex: The component of edm::Ref containing the product ID and product getter and the index into the collection.
    This class is a specialization of RefCore and forwards most of the
    implementation to RefCore.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/refcore_implementation.h"
#include "DataFormats/Common/interface/RefCore.h"

#include <algorithm>
#include <typeinfo>

#include <atomic>

namespace edm {
  class EDProductGetter;
  class WrapperBase;

  class RefCoreWithIndex {
  public:
    RefCoreWithIndex()
        : cachePtr_(nullptr), processIndex_(0), productIndex_(0), elementIndex_(edm::key_traits<unsigned int>::value) {}

    RefCoreWithIndex(ProductID const& theId,
                     void const* prodPtr,
                     EDProductGetter const* prodGetter,
                     bool transient,
                     unsigned int elementIndex);

    RefCoreWithIndex(RefCore const& iCore, unsigned int);

    RefCoreWithIndex(RefCoreWithIndex const&);

    RefCoreWithIndex& operator=(RefCoreWithIndex const&);

    RefCoreWithIndex(RefCoreWithIndex&& iOther) noexcept
        : processIndex_(iOther.processIndex_),
          productIndex_(iOther.productIndex_),
          elementIndex_(iOther.elementIndex_) {
      cachePtr_.store(iOther.cachePtr_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    RefCoreWithIndex& operator=(RefCoreWithIndex&& iOther) noexcept {
      cachePtr_.store(iOther.cachePtr_.load(std::memory_order_relaxed), std::memory_order_relaxed);
      processIndex_ = iOther.processIndex_;
      productIndex_ = iOther.productIndex_;
      elementIndex_ = iOther.elementIndex_;
      return *this;
    }

    ~RefCoreWithIndex() noexcept {}

    ProductID id() const { ID_IMPL; }

    /**If productPtr is not 0 then productGetter will be 0 since only one is available at a time */
    void const* productPtr() const { PRODUCTPTR_IMPL; }

    /**This function is 'const' even though it changes an internal value becuase it is meant to be
     used as a way to store in a thread-safe way a cache of a value. This allows classes which use
     the RefCore to not have to declare it 'mutable'
     */
    void setProductPtr(void const* prodPtr) const { setCacheIsProductPtr(prodPtr); }

    /**This function is 'const' even though it changes an internal value becuase it is meant to be
     used as a way to store in a thread-safe way a cache of a value. This allows classes which use
     the RefCore to not have to declare it 'mutable'
     */
    bool tryToSetProductPtrForFirstTime(void const* prodPtr) const {
      return refcoreimpl::tryToSetCacheItemForFirstTime(cachePtr_, prodPtr);
    }

    unsigned int index() const { return elementIndex_; }

    // Checks for null
    bool isNull() const { return !isNonnull(); }

    // Checks for non-null
    bool isNonnull() const { ISNONNULL_IMPL; }

    // Checks for null
    bool operator!() const { return isNull(); }

    // Checks if collection is in memory or available
    // in the Event. No type checking is done.
    // This function is potentially costly as it might cause a disk
    // read (note that it does not cause the data to be cached locally)
    bool isAvailable() const { return toRefCore().isAvailable(); }

    //Convert to an equivalent RefCore. Needed for Ref specialization.
    RefCore const& toRefCore() const { return *reinterpret_cast<const RefCore*>(this); }

    EDProductGetter const* productGetter() const { PRODUCTGETTER_IMPL; }

    void setProductGetter(EDProductGetter const* prodGetter) const { toRefCore().setProductGetter(prodGetter); }

    WrapperBase const* getProductPtr(std::type_info const& type, EDProductGetter const* prodGetter) const {
      return toRefCore().getProductPtr(type, prodGetter);
    }

    void productNotFoundException(std::type_info const& type) const { toRefCore().productNotFoundException(type); }

    void wrongTypeException(std::type_info const& expectedType, std::type_info const& actualType) const {
      toRefCore().wrongTypeException(expectedType, actualType);
    }

    void nullPointerForTransientException(std::type_info const& type) const {
      toRefCore().nullPointerForTransientException(type);
    }

    void swap(RefCoreWithIndex&);

    bool isTransient() const { ISTRANSIENT_IMPL; }

    int isTransientInt() const { return isTransient() ? 1 : 0; }

    void pushBackItem(RefCoreWithIndex const& productToBeInserted, bool checkPointer) {
      toUnConstRefCore().pushBackItem(productToBeInserted.toRefCore(), checkPointer);
    }

  private:
    RefCore& toUnConstRefCore() { return *reinterpret_cast<RefCore*>(this); }

    void setId(ProductID const& iId) { toUnConstRefCore().setId(iId); }
    void setTransient() { SETTRANSIENT_IMPL; }
    void setCacheIsProductPtr(void const* iItem) const { SETCACHEISPRODUCTPTR_IMPL(iItem); }
    void setCacheIsProductGetter(EDProductGetter const* iGetter) const { SETCACHEISPRODUCTGETTER_IMPL(iGetter); }

    //NOTE: the order MUST remain the same as a RefCore
    // since we play tricks to allow a pointer to a RefCoreWithIndex
    // to be the same as a pointer to a RefCore

    //The low bit of the address is used to determine  if the cachePtr_
    // is storing the productPtr or the EDProductGetter. The bit is set if
    // the address refers to the EDProductGetter.
    mutable std::atomic<void const*> cachePtr_;  // transient
    //The following is what is stored in a ProductID
    // the high bit of processIndex is used to store info on
    // if this is transient.
    ProcessIndex processIndex_;
    ProductIndex productIndex_;
    unsigned int elementIndex_;
  };

  inline void RefCoreWithIndex::swap(RefCoreWithIndex& other) {
    std::swap(processIndex_, other.processIndex_);
    std::swap(productIndex_, other.productIndex_);
    other.cachePtr_.store(cachePtr_.exchange(other.cachePtr_.load()));
    std::swap(elementIndex_, other.elementIndex_);
  }

  inline void swap(edm::RefCoreWithIndex& lhs, edm::RefCoreWithIndex& rhs) { lhs.swap(rhs); }
}  // namespace edm

#endif
