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

// user include files
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/FillViewHelperVector.h"
#include "DataFormats/Common/interface/GetProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/TestHandle.h"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Provenance/interface/ProductID.h"

// system include files
#include <type_traits>

// forward declarations
namespace edm {
  template <typename T>
  class Ptr {
    friend class PtrVectorBase;

  public:
    typedef unsigned long key_type;
    typedef T value_type;

    // General purpose constructor from handle.
    template <typename C>
    Ptr(Handle<C> const& handle, key_type itemKey, bool /*setNow*/ = true)
        : core_(handle.id(), getItem_(handle.product(), itemKey), nullptr, false), key_(itemKey) {}

    // General purpose constructor from orphan handle.
    template <typename C>
    Ptr(OrphanHandle<C> const& handle, key_type itemKey, bool /*setNow*/ = true)
        : core_(handle.id(), getItem_(handle.product(), itemKey), nullptr, false), key_(itemKey) {}

    // General purpose "constructor" from a Ref.
    // Use the conversion function template:
    // ptr = refToPtr(ref)
    // defined in DataFormats/Common/interface/RefToPtr.h
    // to construct a Ptr<T> from a Ref<C>, where T is C::value_type.

    // Constructors for ref to object that is not in an event.
    // An exception will be thrown if an attempt is made to persistify
    // any object containing this Ptr.  Also, in the future work will
    // be done to throw an exception if an attempt is made to put any object
    // containing this Ptr into an event(or run or lumi).

    template <typename C>
    Ptr(C const* iProduct, key_type iItemKey, bool /*setNow*/ = true)
        : core_(ProductID(), iProduct != nullptr ? getItem_(iProduct, iItemKey) : nullptr, nullptr, true),
          key_(iProduct != nullptr ? iItemKey : key_traits<key_type>::value) {}

    Ptr(T const* item, key_type iItemKey)
        : core_(ProductID(), item, nullptr, true), key_(item != nullptr ? iItemKey : key_traits<key_type>::value) {}

    // Constructor from test handle.
    // An exception will be thrown if an attempt is made to persistify
    // any object containing this Ptr.
    template <typename C>
    Ptr(TestHandle<C> const& handle, key_type itemKey, bool /*setNow*/ = true)
        : core_(handle.id(), getItem_(handle.product(), itemKey), nullptr, true), key_(itemKey) {}

    /** Constructor for those users who do not have a product handle,
     but have a pointer to a product getter (such as the EventPrincipal).
     prodGetter will ususally be a pointer to the event principal. */
    Ptr(ProductID const& productID, key_type itemKey, EDProductGetter const* prodGetter)
        : core_(productID, nullptr, mustBeNonZero(prodGetter, "Ptr", productID), false), key_(itemKey) {}

    /** Constructor for use in the various X::fillView(...) functions
     or for extracting a persistent Ptr from a PtrVector.
     It is an error (not diagnosable at compile- or run-time) to call
     this constructor with a pointer to a T unless the pointed-to T
     object is already in a collection of type C stored in the
     Event. The given ProductID must be the id of the collection
     in the Event. */
    Ptr(ProductID const& productID, T const* item, key_type item_key)
        : core_(productID, item, nullptr, false), key_(item_key) {}

    Ptr(ProductID const& productID, T const* item, key_type item_key, bool transient)
        : core_(productID, item, nullptr, transient), key_(item_key) {}

    /** Constructor that creates an invalid ("null") Ptr that is
     associated with a given product (denoted by that product's
     ProductID). */

    explicit Ptr(ProductID const& iId) : core_(iId, nullptr, nullptr, false), key_(key_traits<key_type>::value) {}

    Ptr() : core_(), key_(key_traits<key_type>::value) {}

    template <typename U>
    Ptr(Ptr<U> const& iOther, std::enable_if_t<std::is_base_of<T, U>::value>* = nullptr)
        : core_(iOther.id(),
                (iOther.hasProductCache() ? static_cast<T const*>(iOther.get()) : static_cast<T const*>(nullptr)),
                iOther.productGetter(),
                iOther.isTransient()),
          key_(iOther.key()) {
      //make sure a race condition didn't happen where between the call to hasProductCache() and
      // productGetter() the object was gotten
      if (iOther.hasProductCache() and not hasProductCache()) {
        core_.setProductPtr(static_cast<T const*>(iOther.get()));
      }
    }

    template <typename U>
    explicit Ptr(Ptr<U> const& iOther, std::enable_if_t<std::is_base_of<U, T>::value>* = nullptr)
        : core_(iOther.id(), dynamic_cast<T const*>(iOther.get()), nullptr, iOther.isTransient()), key_(iOther.key()) {}

    /// Destructor
    ~Ptr() {}

    /// Dereference operator
    T const& operator*() const;

    /// Member dereference operator
    T const* operator->() const;

    /// Returns C++ pointer to the item
    T const* get() const { return isNull() ? nullptr : this->operator->(); }

    /// Checks for null
    bool isNull() const { return !isNonnull(); }

    /// Checks for non-null
    //bool isNonnull() const {return id().isValid(); }
    bool isNonnull() const { return key_traits<key_type>::value != key_; }
    /// Checks for null
    bool operator!() const { return isNull(); }

    /// Checks if collection is in memory or available
    /// in the event. No type checking is done.
    bool isAvailable() const;

    /// Checks if this Ptr is transient (i.e. not persistable).
    bool isTransient() const { return core_.isTransient(); }

    /// Accessor for product ID.
    ProductID id() const { return core_.id(); }

    /// Accessor for product getter.
    EDProductGetter const* productGetter() const { return core_.productGetter(); }

    key_type key() const { return key_; }

    bool hasProductCache() const { return nullptr != core_.productPtr(); }

    RefCore const& refCore() const { return core_; }
    // ---------- member functions ---------------------------

    void const* product() const { return nullptr; }

    //Used by ROOT storage
    CMS_CLASS_VERSION(10)

  private:
    template <typename C>
    T const* getItem_(C const* product, key_type iKey);

    void getData_(bool throwIfNotFound = true) const {
      EDProductGetter const* getter = productGetter();
      if (getter != nullptr) {
        WrapperBase const* prod = getter->getIt(core_.id());
        unsigned int iKey = key_;
        if (prod == nullptr) {
          auto optionalProd = getter->getThinnedProduct(core_.id(), key_);
          if (not optionalProd.has_value()) {
            if (throwIfNotFound) {
              core_.productNotFoundException(typeid(T));
            } else {
              return;
            }
          }
          std::tie(prod, iKey) = *optionalProd;
        }
        void const* ad = nullptr;
        prod->setPtr(typeid(T), iKey, ad);
        core_.setProductPtr(ad);
      }
    }
    // ---------- member data --------------------------------
    RefCore core_;
    key_type key_;
  };

  template <typename T>
  template <typename C>
  T const* Ptr<T>::getItem_(C const* iProduct, key_type iKey) {
    assert(iProduct != nullptr);
    typename C::const_iterator it = iProduct->begin();
    std::advance(it, iKey);
    T const* address = detail::GetProduct<C>::address(it);
    return address;
  }

  /// Dereference operator
  template <typename T>
  inline T const& Ptr<T>::operator*() const {
    getData_();
    return *reinterpret_cast<T const*>(core_.productPtr());
  }

  /// Member dereference operator
  template <typename T>
  inline T const* Ptr<T>::operator->() const {
    getData_();
    return reinterpret_cast<T const*>(core_.productPtr());
  }

  template <typename T>
  inline bool Ptr<T>::isAvailable() const {
    getData_(false);
    return hasProductCache();
  }

  template <typename T>
  inline bool operator==(Ptr<T> const& lhs, Ptr<T> const& rhs) {
    return lhs.refCore() == rhs.refCore() && lhs.key() == rhs.key();
  }

  template <typename T>
  inline bool operator!=(Ptr<T> const& lhs, Ptr<T> const& rhs) {
    return !(lhs == rhs);
  }

  template <typename T>
  inline bool operator<(Ptr<T> const& lhs, Ptr<T> const& rhs) {
    /// The ordering of integer keys guarantees that the ordering of Ptrs within
    /// a collection will be identical to the ordering of the referenced objects in the collection.
    return (lhs.refCore() == rhs.refCore() ? lhs.key() < rhs.key() : lhs.refCore() < rhs.refCore());
  }
}  // namespace edm

//The following is needed to get RefToBase to work with an edm::Ptr
//Handle specialization here
#include "DataFormats/Common/interface/HolderToVectorTrait_Ptr_specialization.h"
#include <vector>

namespace edm {
  template <typename T>
  inline void fillView(std::vector<edm::Ptr<T> > const& obj,
                       ProductID const& id,
                       std::vector<void const*>& pointers,
                       FillViewHelperVector& helpers) {
    pointers.reserve(obj.size());
    helpers.reserve(obj.size());
    for (auto const& p : obj) {
      if (p.isAvailable()) {
        pointers.push_back(p.get());
      } else {
        pointers.push_back(nullptr);
      }
      helpers.emplace_back(p.id(), p.key());
    }
  }
}  // namespace edm

#endif
