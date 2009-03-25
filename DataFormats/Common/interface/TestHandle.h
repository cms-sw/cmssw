#ifndef DataFormats_Common_TestHandle_h
#define DataFormats_Common_TestHandle_h

/*----------------------------------------------------------------------
  
TestHandle: Non-owning "smart pointer" for reference to EDProducts.


This is a very preliminary version, and lacks safety features and
elegance.

If the pointed-to EDProduct is destroyed, use of the TestHandle
becomes undefined. There is no way to query the TestHandle to
discover if this has happened.

TestHandles can have:
  -- Product pointer null and id == 0;
  -- Product pointer valid and id != 0;

To check validity, one can use the isValid() function.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/OrphanHandleBase.h"

namespace edm {
  class EDProduct;

  template <typename T>
  class TestHandle : public OrphanHandleBase {
  public:
    typedef T element_type;

    // Default constructed handles are invalid.
    TestHandle();

    TestHandle(T const* prod, ProductID const& id);

    ~TestHandle();

    T const* product() const;
    T const* operator->() const; // alias for product()
    T const& operator*() const;

  private:
  };

  template <class T>
  TestHandle<T>::TestHandle() : OrphanHandleBase()
  { }

  template <class T>
  TestHandle<T>::TestHandle(T const* prod, ProductID const& theId) : OrphanHandleBase(prod, theId) {
  }

  template <class T>
  TestHandle<T>::~TestHandle() {}

  template <class T>
  T const* 
  TestHandle<T>::product() const {
    return static_cast<T const*>(productStorage());
  }

  template <class T>
  T const* 
  TestHandle<T>::operator->() const {
    return product();
  }

  template <class T>
  T const& 
  TestHandle<T>::operator*() const {
    return *product();
  }

}
#endif
