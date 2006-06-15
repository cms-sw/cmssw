#ifndef Framework_OrphanHandle_h
#define Framework_OrphanHandle_h

/*----------------------------------------------------------------------
  
OrphanHandle: Non-owning "smart pointer" for reference to EDProducts.


This is a very preliminary version, and lacks safety features and
elegance.

If the pointed-to EDProduct is destroyed, use of the OrphanHandle
becomes undefined. There is no way to query the OrphanHandle to
discover if this has happened.

OrphanHandles can have:
  -- Product pointer null and id == 0;
  -- Product pointer valid and id != 0;

To check validity, one can use the isValid() function.

$Id: OrphanHandle.h,v 1.1 2006/02/18 06:19:39 wmtan Exp $

----------------------------------------------------------------------*/

#include <algorithm>
#include <stdexcept>
#include <typeinfo>

#include "DataFormats/Common/interface/ProductID.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  class EDProduct;

  template <typename T>
  class OrphanHandle {
  public:
    // Default constructed handles are invalid.
    OrphanHandle();

    OrphanHandle(const OrphanHandle<T>& h);

    OrphanHandle(T const* prod, ProductID const& id);

    ~OrphanHandle();

    void swap(OrphanHandle<T>& other);

    
    OrphanHandle<T>& operator=(const OrphanHandle<T>& rhs);

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
  OrphanHandle<T>::OrphanHandle() :
    prod_(0),
    id_(0)
  { }

  template <class T>
  OrphanHandle<T>::OrphanHandle(const OrphanHandle<T>& h) :
    prod_(h.prod_),
    id_(h.id_)
  { }

  template <class T>
  OrphanHandle<T>::OrphanHandle(T const* product, ProductID const& id) :
    prod_(product),
    id_(id) { 
      assert(prod_);
      assert(id_ != ProductID());
  }

  template <class T>
  OrphanHandle<T>::~OrphanHandle() { 
    // Nothing to do -- we do not own the things to which we point.
  }

  template <class T>
  void
  OrphanHandle<T>::swap(OrphanHandle<T>& other) {
    std::swap(prod_, other.prod_);
    std::swap(id_, other.id_);
  }

  template <class T>
  OrphanHandle<T>&
  OrphanHandle<T>::operator=(const OrphanHandle<T>& rhs) {
    OrphanHandle<T> temp(rhs);
    this->swap(temp);
    return *this;
  }

  template <class T>
  bool
  OrphanHandle<T>::isValid() const {
    return prod_ != 0 && id_ != ProductID();
  }

  template <class T>
  T const* 
  OrphanHandle<T>::product() const {
    // Should we throw if the pointer is null?
    return prod_;
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

  template <class T>
  ProductID 
  OrphanHandle<T>::id() const {
    return id_;
  }
}

#endif
