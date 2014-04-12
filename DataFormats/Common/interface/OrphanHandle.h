#ifndef DataFormats_Common_OrphanHandle_h
#define DataFormats_Common_OrphanHandle_h

/*----------------------------------------------------------------------
  
OrphanHandle: Non-owning "smart pointer" for reference to EDProducts.

This is a very preliminary version, and lacks safety features and elegance.

If the pointed-to EDProduct is destroyed, use of the OrphanHandle
becomes undefined. There is no way to query the OrphanHandle to
discover if this has happened.

OrphanHandles can have:
  -- Product pointer null and id == 0;
  -- Product pointer valid and id != 0;

To check validity, one can use the isValid() function.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/OrphanHandleBase.h"

namespace edm {
  template <typename T>
  class OrphanHandle : public OrphanHandleBase {
  public:
    typedef T element_type;

    // Default constructed handles are invalid.
    OrphanHandle();

    OrphanHandle(T const* prod, ProductID const& id);

    ~OrphanHandle();

    T const* product() const;
    T const* operator->() const; // alias for product()
    T const& operator*() const;

  private:
  };

  template <class T>
  OrphanHandle<T>::OrphanHandle() : OrphanHandleBase()
  { }

  template <class T>
  OrphanHandle<T>::OrphanHandle(T const* prod, ProductID const& theId) : OrphanHandleBase(prod, theId) {
  }

  template <class T>
  OrphanHandle<T>::~OrphanHandle() {}

  template <class T>
  T const* 
  OrphanHandle<T>::product() const {
    return static_cast<T const*>(productStorage());
  }

  template <class T>
  T const* 
  OrphanHandle<T>::operator->() const {
    return product();
  }

  template <class T>
  T const& 
  OrphanHandle<T>::operator*() const {
    return *product();
  }

}
#endif
