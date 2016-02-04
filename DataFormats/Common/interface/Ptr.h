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
//

// system include files
#include "boost/type_traits/is_base_of.hpp"
#include "boost/utility/enable_if.hpp"

// user include files
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/GetProduct.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/TestHandle.h"

// forward declarations
namespace edm {
  template <typename T>
  class Ptr {
     friend class PtrVectorBase;
  public:

    typedef unsigned long key_type;
    typedef T   value_type;

    // General purpose constructor from handle.
    template <typename C>
    Ptr(Handle<C> const& handle, key_type itemKey, bool setNow=true):
    core_(handle.id(), getItem_(handle.product(), itemKey), 0, false), key_(itemKey) {}

    // General purpose constructor from orphan handle.
    template <typename C>
    Ptr(OrphanHandle<C> const& handle, key_type itemKey, bool setNow=true):
    core_(handle.id(), getItem_(handle.product(), itemKey), 0, false), key_(itemKey) {}

    // General purpose "constructor" from a Ref.
    // Use the conversion function template:
    // ptr = refToPtr(ref)
    // defined in DataFormats/Common/interface/RefToPtr.h
    // to construct a Ptr<T> from a Ref<C>, where T is C::value_type.

    // Constructor for ref to object that is not in an event.
    // An exception will be thrown if an attempt is made to persistify
    // any object containing this Ptr.  Also, in the future work will
    // be done to throw an exception if an attempt is made to put any object
    // containing this Ptr into an event(or run or lumi).
    template <typename C>
    Ptr(C const* product, key_type itemKey, bool setNow=true):
    core_(ProductID(), product != 0 ? getItem_(product,itemKey) : 0, 0, true),
	 key_(product != 0 ? itemKey : key_traits<key_type>::value) {}

    // Constructor from test handle.
    // An exception will be thrown if an attempt is made to persistify
    // any object containing this Ptr.
    template <typename C>
    Ptr(TestHandle<C> const& handle, key_type itemKey, bool setNow=true):
    core_(handle.id(), getItem_(handle.product(), itemKey), 0, true), key_(itemKey) {}

    /** Constructor for those users who do not have a product handle,
     but have a pointer to a product getter (such as the EventPrincipal).
     prodGetter will ususally be a pointer to the event principal. */
    Ptr(ProductID const& productID, key_type itemKey, EDProductGetter const* prodGetter) :
    core_(productID, 0, mustBeNonZero(prodGetter, "Ptr", productID), false), key_(itemKey) {
    }

    /** Constructor for use in the various X::fillView(...) functions
     or for extracting a persistent Ptr from a PtrVector.
     It is an error (not diagnosable at compile- or run-time) to call
     this constructor with a pointer to a T unless the pointed-to T
     object is already in a collection of type C stored in the
     Event. The given ProductID must be the id of the collection
     in the Event. */
    Ptr(ProductID const& productID, T const* item, key_type item_key) :
    core_(productID, item, 0, false),
    key_(item_key) {
    }

    /** Constructor that creates an invalid ("null") Ptr that is
     associated with a given product (denoted by that product's
     ProductID). */

    explicit Ptr(ProductID const& id) :
    core_(id, 0, 0, false),
    key_(key_traits<key_type>::value)
    { }

    Ptr():
    core_(),
    key_(key_traits<key_type>::value)
    {}

    Ptr(Ptr<T> const& iOther):
    core_(iOther.core_),
    key_(iOther.key_)
    {}

    template<typename U>
    Ptr(Ptr<U> const& iOther, typename boost::enable_if_c<boost::is_base_of<T, U>::value>::type * = 0):
    core_(iOther.id(),
          (iOther.hasProductCache() ? static_cast<T const*>(iOther.get()): static_cast<T const*>(0)),
          iOther.productGetter(),
	  iOther.isTransient()),
    key_(iOther.key()) {
    }

    template<typename U>
    explicit
    Ptr(Ptr<U> const& iOther, typename boost::enable_if_c<boost::is_base_of<U, T>::value>::type * = 0):
    core_(iOther.id(),
          dynamic_cast<T const*>(iOther.get()),
          0,
	  iOther.isTransient()),
    key_(iOther.key()) {
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
    bool isNonnull() const {return key_traits<key_type>::value != key_;}
    /// Checks for null
    bool operator!() const {return isNull();}

    /// Checks if collection is in memory or available
    /// in the event. No type checking is done.
    bool isAvailable() const {return core_.isAvailable();}

    /// Checks if this Ptr is transient (i.e. not persistable).
    bool isTransient() const {return core_.isTransient();}

    /// Accessor for product ID.
    ProductID id() const {return core_.id();}

    /// Accessor for product getter.
    EDProductGetter const* productGetter() const {return core_.productGetter();}

    key_type key() const {return key_;}

    bool hasProductCache() const { return 0 != core_.productPtr(); }

    RefCore const& refCore() const {return core_;}
    // ---------- member functions ---------------------------

    void const* product() const {return 0;}
    
    //Used by ROOT storage
    CMS_CLASS_VERSION(10)

  private:
    //Ptr(Ptr const&); // stop default

    /** Constructor for extracting a transient Ptr from a PtrVector. */
    Ptr(T const* item, key_type item_key) :
    core_(ProductID(), item, 0, true),
    key_(item_key) {
    }

    //Ptr const& operator=(Ptr const&); // stop default
    template<typename C>
    T const* getItem_(C const* product, key_type iKey);

    void getData_() const {
      if(!hasProductCache() && 0 != productGetter()) {
        void const* ad = 0;
        EDProduct const* prod = productGetter()->getIt(core_.id());
        if(prod == 0) {
	  core_.productNotFoundException(typeid(T));
        }
        prod->setPtr(typeid(T), key_, ad);
        core_.setProductPtr(ad);
      }
    }
    // ---------- member data --------------------------------
    RefCore core_;
    key_type key_;
  };

  template<typename T>
  template<typename C>
  T const* Ptr<T>::getItem_(C const* product, key_type iKey) {
    assert (product != 0);
    typename C::const_iterator it = product->begin();
    std::advance(it,iKey);
    T const* address = detail::GetProduct<C>::address(it);
    return address;
  }

  /// Dereference operator
  template <typename T>
  inline
  T const&
  Ptr<T>::operator*() const {
    getData_();
    return *reinterpret_cast<T const*>(core_.productPtr());
  }

  /// Member dereference operator
  template <typename T>
  inline
  T const*
  Ptr<T>::operator->() const {
    getData_();
    return reinterpret_cast<T const*>(core_.productPtr());
  }

  template <typename T>
  inline
  bool
  operator==(Ptr<T> const& lhs, Ptr<T> const& rhs) {
    return lhs.refCore() == rhs.refCore() && lhs.key() == rhs.key();
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
    /// The ordering of integer keys guarantees that the ordering of Ptrs within
    /// a collection will be identical to the ordering of the referenced objects in the collection.
    return (lhs.refCore() == rhs.refCore() ? lhs.key() < rhs.key() : lhs.refCore() < rhs.refCore());
  }

}

#endif
