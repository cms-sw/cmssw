#ifndef DataFormats_Common_ValidHandle_h
#define DataFormats_Common_ValidHandle_h

/*----------------------------------------------------------------------
  
Handle: Non-owning "smart pointer" for reference to products and
their provenances.

The data product is always guaranteed to be valid for this handle type.
----------------------------------------------------------------------*/

#include <utility>
#include "DataFormats/Provenance/interface/ProductID.h"

namespace edm {
  namespace vhhelper {
    void throwIfNotValid(const void*) noexcept(false);
  }
  template <typename T>
  class ValidHandle {
  public:
    using element_type = T;

    ValidHandle() = delete;
    ValidHandle(T const* prod, ProductID id) noexcept(false) : product_(prod), id_(id) {
      vhhelper::throwIfNotValid(prod);
    }

    //NOTE: C++ disallows references to null
    ValidHandle(T const& prod, ProductID id) noexcept(true) : product_(&prod), id_(id) {}
    ValidHandle(const ValidHandle<T>&) = default;
    ValidHandle<T>& operator=(ValidHandle<T> const& rhs) = default;
    ~ValidHandle() = default;

    ProductID const& id() const noexcept(true) { return id_; }

    T const* product() const noexcept(true) { return product_; }

    T const* operator->() const noexcept(true) { return product(); }
    T const& operator*() const noexcept(true) { return *product(); }

  private:
    T const* product_;
    ProductID id_;
  };

  /** Take a handle (e.g. edm::Handle<T> or edm::OrphanHandle<T> and
   create a edm::ValidHandle<T>. If the argument is an invalid handle,
   an exception will be thrown.
   */
  template <typename U>
  auto makeValid(const U& iOtherHandleType) noexcept(false) {
    vhhelper::throwIfNotValid(iOtherHandleType.product());
    //because of the check, we know this is valid and do not have to check again
    return ValidHandle<typename U::element_type>(*iOtherHandleType.product(), iOtherHandleType.id());
  }
}  // namespace edm

#endif
