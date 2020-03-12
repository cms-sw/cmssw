#ifndef DataFormats_Common_Handle_h
#define DataFormats_Common_Handle_h

/*----------------------------------------------------------------------
  
Handle: Non-owning "smart pointer" for reference to Products and
their Provenances.

This is a very preliminary version, and lacks safety features and
elegance.

If the pointed-to Product or Provenance is destroyed, use of the
Handle becomes undefined. There is no way to query the Handle to
discover if this has happened.

Handles can have:
  -- Product and Provenance pointers both null;
  -- Both pointers valid

To check validity, one can use the isValid() function.

If failedToGet() returns true then the requested data is not available
If failedToGet() returns false but isValid() is also false then no attempt 
  to get data has occurred

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/HandleBase.h"

namespace edm {

  template <typename T>
  class Handle : public HandleBase {
  public:
    using element_type = T;

    // Default constructed handles are invalid.
    Handle();

    Handle(T const* prod, Provenance const* prov);

    Handle(std::shared_ptr<HandleExceptionFactory const>&&);
    Handle(Handle const&) = default;
    Handle& operator=(Handle&&) = default;
    Handle& operator=(Handle const&) = default;

    ~Handle();

    T const* product() const;
    T const* operator->() const;  // alias for product()
    T const& operator*() const;

  private:
  };

  template <class T>
  Handle<T>::Handle() : HandleBase() {}

  template <class T>
  Handle<T>::Handle(T const* prod, Provenance const* prov) : HandleBase(prod, prov) {}

  template <class T>
  Handle<T>::Handle(std::shared_ptr<edm::HandleExceptionFactory const>&& iWhyFailed)
      : HandleBase(std::move(iWhyFailed)) {}

  template <class T>
  Handle<T>::~Handle() {}

  template <class T>
  T const* Handle<T>::product() const {
    return static_cast<T const*>(productStorage());
  }

  template <class T>
  T const* Handle<T>::operator->() const {
    return product();
  }

  template <class T>
  T const& Handle<T>::operator*() const {
    return *product();
  }
}  // namespace edm
#endif
