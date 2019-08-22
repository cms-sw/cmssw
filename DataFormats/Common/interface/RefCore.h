#ifndef DataFormats_Common_RefCore_h
#define DataFormats_Common_RefCore_h

/*----------------------------------------------------------------------
  
RefCore: The component of edm::Ref containing the product ID and product getter.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/refcore_implementation.h"

#include <algorithm>
#include <typeinfo>
#include <atomic>

namespace edm {
  class RefCoreWithIndex;
  class EDProductGetter;
  class WrapperBase;

  class RefCore {
    //RefCoreWithIndex is a specialization of RefCore done for performance
    // Since we need to freely convert one to the other the friendship is used
    friend class RefCoreWithIndex;

  public:
    RefCore() : cachePtr_(nullptr), processIndex_(0), productIndex_(0) {}

    RefCore(ProductID const& theId, void const* prodPtr, EDProductGetter const* prodGetter, bool transient);

    RefCore(RefCore const&);

    RefCore& operator=(RefCore const&);

    RefCore(RefCore&& iOther) noexcept : processIndex_(iOther.processIndex_), productIndex_(iOther.productIndex_) {
      cachePtr_.store(iOther.cachePtr_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    RefCore& operator=(RefCore&& iOther) noexcept {
      cachePtr_.store(iOther.cachePtr_.load(std::memory_order_relaxed), std::memory_order_relaxed);
      processIndex_ = iOther.processIndex_;
      productIndex_ = iOther.productIndex_;
      return *this;
    }

    ~RefCore() noexcept {}

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

    // Checks for null
    bool isNull() const { return !isNonnull(); }

    // Checks for non-null
    bool isNonnull() const { ISNONNULL_IMPL; }

    // Checks for null
    bool operator!() const { return isNull(); }

    // Checks if collection is in memory or available
    // in the Event. No type checking is done.

    bool isAvailable() const;

    EDProductGetter const* productGetter() const { PRODUCTGETTER_IMPL; }

    void setProductGetter(EDProductGetter const* prodGetter) const;

    WrapperBase const* getProductPtr(std::type_info const& type, EDProductGetter const* prodGetter) const;

    WrapperBase const* tryToGetProductPtr(std::type_info const& type, EDProductGetter const* prodGetter) const;

    WrapperBase const* getThinnedProductPtr(std::type_info const& type,
                                            unsigned int& thinnedKey,
                                            EDProductGetter const* prodGetter) const;

    bool isThinnedAvailable(unsigned int thinnedKey, EDProductGetter const* prodGetter) const;

    void productNotFoundException(std::type_info const& type) const;

    void wrongTypeException(std::type_info const& expectedType, std::type_info const& actualType) const;

    void nullPointerForTransientException(std::type_info const& type) const;

    void swap(RefCore&) noexcept;

    bool isTransient() const { ISTRANSIENT_IMPL; }

    int isTransientInt() const { return isTransient() ? 1 : 0; }

    void pushBackItem(RefCore const& productToBeInserted, bool checkPointer);

    void pushBackRefItem(RefCore const& productToBeInserted);

  private:
    RefCore(void const* iCache, ProcessIndex iProcessIndex, ProductIndex iProductIndex)
        : cachePtr_(iCache), processIndex_(iProcessIndex), productIndex_(iProductIndex) {}
    void setId(ProductID const& iId);
    void setTransient() { SETTRANSIENT_IMPL; }
    void setCacheIsProductPtr(const void* iItem) const { SETCACHEISPRODUCTPTR_IMPL(iItem); }
    void setCacheIsProductGetter(EDProductGetter const* iGetter) const { SETCACHEISPRODUCTGETTER_IMPL(iGetter); }
    bool cachePtrIsInvalid() const {
      return 0 == (reinterpret_cast<std::uintptr_t>(cachePtr_.load()) & refcoreimpl::kCacheIsProductPtrMask);
    }

    //The low bit of the address is used to determine  if the cachePtr_
    // is storing the productPtr or the EDProductGetter. The bit is set if
    // the address refers to the EDProductGetter.
    mutable std::atomic<void const*> cachePtr_;  // transient
    //The following is what is stored in a ProductID
    // the high bit of processIndex is used to store info on
    // if this is transient.
    //If the type or order of the member data is changes you MUST also update
    // the custom streamer in RefCoreStreamer.cc and RefCoreWithIndex
    ProcessIndex processIndex_;
    ProductIndex productIndex_;
  };

  inline bool operator==(RefCore const& lhs, RefCore const& rhs) {
    return lhs.isTransient() == rhs.isTransient() &&
           (lhs.isTransient() ? lhs.productPtr() == rhs.productPtr() : lhs.id() == rhs.id());
  }

  inline bool operator!=(RefCore const& lhs, RefCore const& rhs) { return !(lhs == rhs); }

  inline bool operator<(RefCore const& lhs, RefCore const& rhs) {
    return lhs.isTransient() ? (rhs.isTransient() ? lhs.productPtr() < rhs.productPtr() : false)
                             : (rhs.isTransient() ? true : lhs.id() < rhs.id());
  }

  inline void RefCore::swap(RefCore& other) noexcept {
    std::swap(processIndex_, other.processIndex_);
    std::swap(productIndex_, other.productIndex_);
    other.cachePtr_.store(cachePtr_.exchange(other.cachePtr_.load()));
  }

  inline void swap(edm::RefCore& lhs, edm::RefCore& rhs) { lhs.swap(rhs); }
}  // namespace edm

#endif
