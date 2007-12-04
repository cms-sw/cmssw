#ifndef DataFormats_Common_Ptr_h
#define DataFormats_Common_Ptr_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     Ptr
// 
/**\class edm::Ptr Ptr.h DataFormats/Common/interface/Ptr.h

 Description: Persistent 'pointer' to an item in a collection where the collection is in the edm::Event

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Oct 18 14:41:33 CEST 2007
// $Id: Ptr.h,v 1.2 2007/10/31 18:51:02 chrjones Exp $
//

// system include files
#include "boost/static_assert.hpp"
#include "boost/type_traits/is_base_of.hpp"

// user include files
#include "FWCore/Utilities/interface/GCCPrerequisite.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/GetProduct.h"
#include "DataFormats/Common/interface/EDProduct.h"

// forward declarations
namespace edm {
  template <typename T>
  class Ptr {
      
  public:
    
    typedef unsigned long key_type;
    typedef T   value_type;
    /** General purpose constructor from handle like object.
     id(), returning a ProductID
     product(), returning a C*. */
    
    template <typename HandleC>
    Ptr(HandleC const& handle, key_type itemKey, bool setNow=true):
    core_(handle.id(),getItem_(handle,itemKey),0),key_(itemKey) {}
    
    /** Constructor for those users who do not have a product handle,
     but have a pointer to a product getter (such as the EventPrincipal).
     prodGetter will ususally be a pointer to the event principal. */
    Ptr(ProductID const& productID, key_type itemKey, EDProductGetter const* prodGetter) :
    core_(productID, 0, prodGetter), key_(itemKey) {
    }
    
    /** Constructor for use in the various X::fillView(...) functions.
     It is an error (not diagnosable at compile- or run-time) to call
     this constructor with a pointer to a T unless the pointed-to T
     object is already in a collection of type C stored in the
     Event. The given ProductID must be the id of the collection in
     the Event. */
    Ptr(ProductID const& productID, T const* item, key_type item_key) :
    core_(productID, item, 0),
    key_(item_key)
    { 
    }
  
    /** Constructor that creates an invalid ("null") Ptr that is
     associated with a given product (denoted by that product's
     ProductID). */
    
    explicit Ptr(ProductID const& id) :
    core_(id, 0, 0),
    key_(key_traits<key_type>::value)
    { }
    
    Ptr():
    core_(),
    key_(key_traits<key_type>::value)
    {}
    
    Ptr(const Ptr<T>& iOther):
    core_(iOther.core_),
    key_(iOther.key_)
    {}

    template< typename U>
    Ptr(const Ptr<U>& iOther):
    core_(iOther.id(), 
          (iOther.hasCache()? static_cast<const T*>(iOther.get()): static_cast<const T*>(0)),
          iOther.productGetter()),
    key_(iOther.key())
    {
      //check that types are assignable
      BOOST_STATIC_ASSERT( (boost::is_base_of<T, U>::value) );
    }
    
    /// Destructor
    ~Ptr() {}
    
    /// Dereference operator
    T const&
    operator*() const;
    
    /// Member dereference operator
    T const*
    operator->() const;
    
    /// Returns C++ pointer to the item
    T const* get() const {
      return isNull() ? 0 : this->operator->();
    }
    
    /// Checks for null
    bool isNull() const {return !isNonnull(); }
    
    /// Checks for non-null
    //bool isNonnull() const {return id().isValid(); }
    bool isNonnull() const { return core_.isNonnull(); }
    
    /// Checks for null
    bool operator!() const {return isNull();}
    
    /// Checks if collection is in memory or available
    /// in the event. No type checking is done.
    bool isAvailable() const {return core_.isAvailable();}

    /// Accessor for product ID.
    ProductID id() const {return core_.id();}
    
    /// Accessor for product getter.
    EDProductGetter const* productGetter() const {return core_.productGetter();}
    
    key_type key() const {return key_;}
    
    bool hasCache() const { return 0!=core_.productPtr(); }
    // ---------- member functions ---------------------------
    
  private:
    //Ptr(const Ptr&); // stop default
    
    //const Ptr& operator=(const Ptr&); // stop default
    template<typename HandleC>
    const T* getItem_(const HandleC& iHandle, key_type iKey);
    
    void getData_() const { 
      if( !hasCache() && 0 != productGetter() ) {
        const void* ad = 0;
        productGetter()->getIt(core_.id())->setPtr(typeid(T),
                                                   key_,
                                                   ad);
        core_.setProductPtr(ad);
      }
    }
    // ---------- member data --------------------------------
    RefCore core_;
    key_type key_;
  };
  
  template<typename T>
  template<typename HandleC>
  const T* Ptr<T>::getItem_(const HandleC& iHandle, key_type iKey)
  {
    typedef typename HandleC::element_type container_type;
    typename container_type::const_iterator it = iHandle.product()->begin();
    advance(it,iKey);
    T const* address = detail::GetProduct<container_type>::address( it );
    return address;
    
  }
  
  /// Dereference operator
  template <typename T>
  inline
  T const&
  Ptr<T>::operator*() const {
    getData_();
    return *reinterpret_cast<const T*>(core_.productPtr());
  }
  
  /// Member dereference operator
  template <typename T>
  inline
  T const*
  Ptr<T>::operator->() const {
    getData_();
    return reinterpret_cast<const T*>(core_.productPtr());
  }
  
  template <typename T>
  inline
  bool
  operator==(Ptr<T> const& lhs, Ptr<T> const& rhs) {
    return lhs.id() == rhs.id() && 
    lhs.key() == rhs.key();
  }
  
  template <typename T>
  inline
  bool
  operator!=(Ptr<T> const& lhs, Ptr<T> const& rhs) {
    return !(lhs == rhs);
  }
  
  template <typename T>
  inline
  bool
  operator<(Ptr<T> const& lhs, Ptr<T> const& rhs) {
#if ! GCC_PREREQUISITE(3,4,4)
    // needed for gcc 3_2_3 compiler bug workaround
    using GCC_3_2_3_WORKAROUND_1::compare_key;
    using GCC_3_2_3_WORKAROUND_2::compare_key;
#endif
    /// the definition and use of compare_key<> guarantees that the ordering of Ptrs within
    /// a collection will be identical to the ordering of the referenced objects in the collection.
    return (lhs.id() == rhs.id() ? lhs.key()< rhs.key() : lhs.id() < rhs.id());
  }
  
}

#endif
