#ifndef Common_TestHandle_h
#define Common_TestHandle_h

/*----------------------------------------------------------------------

$Id: TestHandle.h,v 1.1.4.2 2008/12/11 07:07:12 wmtan Exp $

Version of Handle

----------------------------------------------------------------------*/

#include <algorithm>
#include <stdexcept>
#include <typeinfo>

#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  template <class T>
  class TestHandle {
  public:
    typedef T element_type;
    // Default constructed handles are invalid.
    TestHandle();

    TestHandle(TestHandle<T> const& h);

    TestHandle(T const* prod, ProductID const& id);

    ~TestHandle();

    void swap(TestHandle<T>& other);

    TestHandle<T>& operator=(TestHandle<T> const& rhs);

    bool isValid() const;

    T const* product() const;
    T const* operator->() const; // alias for product()
    T const& operator*() const;

    ProductID id() const;

  private:
    T const* prod_;
    ProductID id_;
  };

  template <class T>
  TestHandle<T>::TestHandle() :
    prod_(0),
    id_()
  { }

  template <class T>
  TestHandle<T>::TestHandle(TestHandle<T> const& h) :
    prod_(h.prod_),
    id_(h.id_)
  { }

  template <class T>
  TestHandle<T>::TestHandle(T const* theProduct, ProductID const& theId) :
    prod_(theProduct),
    id_(theId) {
  }

  template <class T>
  TestHandle<T>::~TestHandle() {
    // Nothing to do -- we do not own the things to which we point.
  }

  template <class T>
  void
  TestHandle<T>::swap(TestHandle<T>& other) {
    std::swap(prod_, other.prod_);
    std::swap(id_, other.id_);
  }

  template <class T>
  TestHandle<T>&
  TestHandle<T>::operator=(TestHandle<T> const& rhs) {
    TestHandle<T> temp(rhs);
    this->swap(temp);
    return *this;
  }

  template <class T>
  bool
  TestHandle<T>::isValid() const {
    return prod_ != 0 && id_ != ProductID();
  }

  template <class T>
  T const*
  TestHandle<T>::product() const {
    return prod_;
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

  template <class T>
  ProductID
  TestHandle<T>::id() const {
    return id_;
  }

  // Free swap function
  template <class T>
  inline
  void
  swap(TestHandle<T>& a, TestHandle<T>& b)
  {
    a.swap(b);
  }
}

#endif
