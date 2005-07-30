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

$Id: Handle.h,v 1.9 2005/07/21 18:26:03 wmtan Exp $

----------------------------------------------------------------------*/

#include <algorithm>
#include <stdexcept>
#include <typeinfo>

#include "boost/utility/enable_if.hpp"
#include "boost/type_traits.hpp"

#include "FWCore/EDProduct/interface/ProductID.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/BasicHandle.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm
{
  template <class T>
  class Handle
  {
  public:
    // Default constructed handles are invalid.
    Handle();

    Handle(const Handle<T>& h);

    Handle(T const* prod, Provenance const* prov, ProductID id);

    ~Handle();

    void swap(Handle<T>& other);

    
    Handle<T>& operator=(const Handle<T>& rhs);

    bool isValid() const;

    T const* product() const;
    T const* operator->() const; // alias for product()
    T const& operator*() const;

    Provenance const* provenance() const;

    ProductID id() const;

  private:
    T const* prod_;
    Provenance const* prov_;    
    ProductID id_;
  };

  template <class T>
  Handle<T>::Handle() :
    prod_(0),
    prov_(0),
    id_(0)
  { }

  template <class T>
  Handle<T>::Handle(const Handle<T>& h) :
    prod_(h.prod_),
    prov_(h.prov_),
    id_(h.id_)
  { }

  template <class T>
  Handle<T>::Handle(T const* product, Provenance const* prov, ProductID id) :
    prod_(product),
    prov_(prov),
    id_(id)
    { 
      assert(prod_);
      assert(prov_);
      assert(id_ != ProductID());
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
    std::swap(id_, other.id_);
  }

  template <class T>
  Handle<T>&
  Handle<T>::operator=(const Handle<T>& rhs)
  {
    Handle<T> temp(rhs);
    this->swap(temp);
    return *this;
  }

  template <class T>
  bool
  Handle<T>::isValid() const
  {
    return prod_ != 0 && prov_ != 0;
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

  template <class T>
  ProductID 
  Handle<T>::id() const
  {
    return id_;
  }

  //------------------------------------------------------------
  // Non-member functions
  //

  // Convert from handle-to-EDProduct to handle-to-T
  template <class T>
  void convert_handle(BasicHandle const& orig,
		      Handle<T>& result)
  {
    EDProduct const* originalWrap = orig.wrapper();
    if (originalWrap == 0)
      throw edm::Exception(edm::errors::InvalidReference,"NullPointer")
      << "edm::BasicHandle has null pointer to Wrapper";
    Wrapper<T> const* wrap = dynamic_cast<Wrapper<T> const*>(originalWrap);
    if (wrap == 0)
      throw edm::Exception(edm::errors::LogicError,"ConvertType")
      << "edm::Wrapper converting from EDProduct to "
      << typeid(originalWrap).name();

    Handle<T> h(wrap->product(), orig.provenance(), wrap->id());
    h.swap(result);
  }

}

#endif
