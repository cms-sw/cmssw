#ifndef EDM_HANDLE_H
#define EDM_HANDLE_H

/*----------------------------------------------------------------------
  
Handle: Non-owning "smart pointer" for reference to EDProducts and
their Provenances.

This is a very preliminary version, and lacks safety features and
elegance.

If the pointed-to EDProduct or Provenance is destroyed, use of the
Handle becomes undefined. There is no way to query the Handle to
discover if this has happened.

Handles can have:
  -- Product and Provenance pointers both null;
  -- Both pointers valid

To check validity, one can use the isValid() function.

$Id: Handle.h,v 1.4 2005/05/03 19:27:52 wmtan Exp $

----------------------------------------------------------------------*/

#include <algorithm>
#include <stdexcept>

#include "boost/utility/enable_if.hpp"
#include "boost/type_traits.hpp"

#include "FWCore/CoreFramework/interface/CoreFrameworkfwd.h"
#include "FWCore/EDProduct/interface/EDProduct.h"

namespace edm
{
  template <class T>
  class Handle
  {
  public:
    // Default constructed handles are invalid.
    Handle();

    Handle(const Handle<T>& h);

    Handle(T const* prod, Provenance const* prov);

    ~Handle();

    void swap(Handle<T>& other);

    
    Handle<T>& operator= (const Handle<T>& rhs);

    // The following code is how I thought disable_if should be used ...
    // but this use fails, for reasons unknown to me.

    // disable_if is used to force the compiler to ignore this
    // template if the template parameter T is EDProduct.
    //     typename boost::disable_if_c<boost::is_same<T, EDProduct>::value
    //                                 , Handle<T>&
    //                                 >::type
    //     operator= (const Handle<EDProduct>& rhs);

    bool isValid() const;

    T const* product() const;
    T const* operator->() const; // alias for product()
    T const& operator*() const;

    Provenance const* provenance() const;

  private:
    T const*          prod_;
    Provenance const* prov_;    
  };

  template <class T>
  Handle<T>::Handle() :
    prod_(0),
    prov_(0)
  { }

  template <class T>
  Handle<T>::Handle(const Handle<T>& h) :
    prod_(h.prod_),
    prov_(h.prov_)
  { }

  template <class T>
  Handle<T>::Handle(T const* product, Provenance const* prov) :
    prod_(product),
    prov_(prov)
    { 
      assert(prod_);
      assert(prov_);
    }

  template <class T>
  Handle<T>::~Handle()
  { 
    // Nothing to do -- we do not own the things to which we point.
  }

  template <class T>
  void
  Handle<T>::swap(Handle<T>& other)
  {
    std::swap(prod_, other.prod_);
    std::swap(prov_, other.prov_);
  }

  template <class T>
  Handle<T>&
  Handle<T>::operator= (const Handle<T>& rhs)
  {
    Handle<T> temp(rhs);
    this->swap(temp);
    return *this;
  }

  template <class T>
  bool
  Handle<T>::isValid() const
  {
    return prod_ && prov_;
  }

  template <class T>
  T const* 
  Handle<T>::product() const
  {
    // Should we throw if the pointer is null?
    return prod_;
  }

  template <class T>
  T const* 
  Handle<T>::operator->() const
  {
    return product();
  }

  template <class T>
  T const& 
  Handle<T>::operator*() const
  {
    return *product();
  }

  template <class T>
  Provenance const* 
  Handle<T>::provenance() const
  {
    // Should we throw if the pointer is null?
    return prov_;
  }

  //------------------------------------------------------------
  // Non-member functions
  //

  // Convert from handle-to-EDProduct to handle-to-T, if the dynamic
  // type of the EDProduct is T. Throw and exception if the type does
  // not match.
  template <class T>
  void convert_handle(const Handle<EDProduct>& orig,
		      Handle<T>& result)
  {
    const T* prod = dynamic_cast<const T*>(orig.product());
    if (prod == 0)  throw std::runtime_error("failed type conversion");
    Handle<T> h(prod, orig.provenance());
    h.swap(result);
  }

}

#endif
