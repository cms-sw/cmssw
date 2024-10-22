#ifndef DataFormats_Common_RefToBaseProd_h
#define DataFormats_Common_RefToBaseProd_h

/* \class edm::RefToBaseProd<T>
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Common/interface/FillView.h"
#include "DataFormats/Common/interface/FillViewHelperVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <utility>
#include <vector>

namespace edm {

  template <typename T>
  class RefToBaseProd {
  public:
    typedef View<T> product_type;

    /// Default constructor needed for reading from persistent store. Not for direct use.
    RefToBaseProd() : product_() {}

    template <typename C>
    explicit RefToBaseProd(Handle<C> const& handle);
    explicit RefToBaseProd(Handle<View<T>> const& handle);
    template <typename C>
    explicit RefToBaseProd(OrphanHandle<C> const& handle);
    RefToBaseProd(const RefToBaseProd<T>&);
    template <typename C>
    explicit RefToBaseProd(const RefProd<C>&);
    RefToBaseProd(ProductID const&, EDProductGetter const*);

    /// Destructor
    ~RefToBaseProd() { delete viewPtr(); }

    /// Dereference operator
    product_type const& operator*() const;

    /// Member dereference operator
    product_type const* operator->() const;

    /// Returns C++ pointer to the product
    /// Will attempt to retrieve product
    product_type const* get() const { return isNull() ? nullptr : this->operator->(); }

    /// Returns C++ pointer to the product
    /// Will attempt to retrieve product
    product_type const* product() const { return isNull() ? 0 : this->operator->(); }

    /// Checks for null
    bool isNull() const { return !isNonnull(); }

    /// Checks for non-null
    bool isNonnull() const { return product_.isNonnull(); }

    /// Checks if collection is in memory or available
    /// in the Event. No type checking is done.
    /// This function is potentially costly as it might cause a disk
    /// read (note that it does not cause the data to be cached locally)
    bool isAvailable() const { return product_.isAvailable(); }

    /// Checks for null
    bool operator!() const { return isNull(); }

    /// Accessor for product ID.
    ProductID id() const { return product_.id(); }

    /// Accessor for product getter.
    EDProductGetter const* productGetter() const { return product_.productGetter(); }

    /// Checks if product is in memory.
    bool hasCache() const { return product_.productPtr() != nullptr; }

    RefToBaseProd<T>& operator=(const RefToBaseProd<T>& other);

    void swap(RefToBaseProd<T>&);

    //Needed for ROOT storage
    CMS_CLASS_VERSION(10)
  private:
    //NOTE: Access to RefCore should be private since we modify the use of productPtr
    RefCore const& refCore() const { return product_; }

    View<T> const* viewPtr() const { return reinterpret_cast<const View<T>*>(product_.productPtr()); }

    RefCore product_;
  };

  template <typename T>
  inline RefToBaseProd<T>::RefToBaseProd(Handle<View<T>> const& handle)
      : product_(handle.id(), nullptr, nullptr, false) {
    product_.setProductPtr(new View<T>(*handle));
  }

  template <typename T>
  inline RefToBaseProd<T>::RefToBaseProd(const RefToBaseProd<T>& ref) : product_(ref.product_) {
    if (product_.productPtr()) {
      product_.setProductPtr(ref.viewPtr() ? (new View<T>(*ref)) : nullptr);
    }
  }

  template <typename T>
  inline RefToBaseProd<T>& RefToBaseProd<T>::operator=(const RefToBaseProd<T>& other) {
    RefToBaseProd<T> temp(other);
    this->swap(temp);
    return *this;
  }

  /// Dereference operator
  template <typename T>
  inline View<T> const& RefToBaseProd<T>::operator*() const {
    return *operator->();
  }

  /// Member dereference operator
  template <typename T>
  inline View<T> const* RefToBaseProd<T>::operator->() const {
    //Another thread might change the value returned so just get it once
    auto getter = product_.productGetter();
    if (getter != nullptr) {
      if (product_.isNull()) {
        Exception::throwThis(errors::InvalidReference, "attempting get view from a null RefToBaseProd.\n");
      }
      ProductID tId = product_.id();
      std::vector<void const*> pointers;
      FillViewHelperVector helpers;
      WrapperBase const* prod = getter->getIt(tId);
      if (prod == nullptr) {
        Exception::throwThis(errors::InvalidReference, "attempting to get view from an unavailable RefToBaseProd.");
      }
      prod->fillView(tId, pointers, helpers);
      std::unique_ptr<View<T>> tmp{new View<T>(pointers, helpers, getter)};
      if (product_.tryToSetProductPtrForFirstTime(tmp.get())) {
        tmp.release();
      }
    }
    return viewPtr();
  }

  template <typename T>
  inline void RefToBaseProd<T>::swap(RefToBaseProd<T>& other) {
    edm::swap(product_, other.product_);
  }

  template <typename T>
  inline bool operator==(RefToBaseProd<T> const& lhs, RefToBaseProd<T> const& rhs) {
    return lhs.refCore() == rhs.refCore();
  }

  template <typename T>
  inline bool operator!=(RefToBaseProd<T> const& lhs, RefToBaseProd<T> const& rhs) {
    return !(lhs == rhs);
  }

  template <typename T>
  inline bool operator<(RefToBaseProd<T> const& lhs, RefToBaseProd<T> const& rhs) {
    return (lhs.refCore() < rhs.refCore());
  }

  template <typename T>
  inline void swap(edm::RefToBaseProd<T> const& lhs, edm::RefToBaseProd<T> const& rhs) {
    lhs.swap(rhs);
  }
}  // namespace edm

#include "DataFormats/Common/interface/FillView.h"

namespace edm {
  template <typename T>
  template <typename C>
  inline RefToBaseProd<T>::RefToBaseProd(const RefProd<C>& ref) : product_(ref.refCore()) {
    std::vector<void const*> pointers;
    FillViewHelperVector helpers;
    fillView(*ref.product(), ref.id(), pointers, helpers);
    product_.setProductPtr(new View<T>(pointers, helpers, ref.refCore().productGetter()));
  }

  template <typename T>
  template <typename C>
  inline RefToBaseProd<T>::RefToBaseProd(Handle<C> const& handle)
      : product_(handle.id(), handle.product(), nullptr, false) {
    std::vector<void const*> pointers;
    FillViewHelperVector helpers;
    fillView(*handle, handle.id(), pointers, helpers);
    product_.setProductPtr(new View<T>(pointers, helpers, nullptr));
  }

  template <typename T>
  template <typename C>
  inline RefToBaseProd<T>::RefToBaseProd(OrphanHandle<C> const& handle)
      : product_(handle.id(), handle.product(), nullptr, false) {
    std::vector<void const*> pointers;
    FillViewHelperVector helpers;
    fillView(*handle, handle.id(), pointers, helpers);
    product_.setProductPtr(new View<T>(pointers, helpers, nullptr));
  }

  template <typename T>
  inline RefToBaseProd<T>::RefToBaseProd(ProductID const& id, EDProductGetter const* getter)
      : product_(id, nullptr, getter, false) {}
}  // namespace edm

#endif
