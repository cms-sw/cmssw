#ifndef DataFormats_Common_FwdPtr_h
#define DataFormats_Common_FwdPtr_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     FwdPtr
//
/**\class edm::FwdPtr FwdPtr.h DataFormats/Common/interface/FwdPtr.h

 Description: Persistent 'pointer' to an item in a collection where the collection is in the edm::Event

 Usage:
    <usage>

*/
//
// Original Author:  Salvatore Rappoccio
//         Created:  Fri Feb  5 14:58:49 CST 2010
//

// user include files
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/GetProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/TestHandle.h"
#include "DataFormats/Common/interface/traits.h"

#include "FWCore/Utilities/interface/EDMException.h"

// system include files
#include "boost/type_traits/is_base_of.hpp"
#include "boost/utility/enable_if.hpp"

// forward declarations
namespace edm {
  template<typename T>
  class FwdPtr {
    friend class FwdPtrVectorBase;
  public:
    
    typedef unsigned long key_type;
    typedef T   value_type;
    
    // General purpose constructor from two Ptrs.
    template<typename C>
    FwdPtr(const Ptr<C>& f, const Ptr<C>& b):
    ptr_(f), backPtr_(b)
    {}
    
  FwdPtr() :
    ptr_(), backPtr_()
      {}
    
    /*     template<typename U> */
    /*     FwdPtr(FwdPtr<U> const& iOther, typename boost::enable_if_c<boost::is_base_of<T, U>::value>::type * t = 0): */
    /*     ptr_(iOther.ptr_, t), backPtr_(iOther.backPtr_,t) */
    /*     { */
    /*     } */
    
    /*     template<typename U> */
    /*     explicit */
    /*     FwdPtr(FwdPtr<U> const& iOther, typename boost::enable_if_c<boost::is_base_of<U, T>::value>::type * t = 0): */
    /*     ptr_(iOther.ptr_,t), backPtr_(iOther.backPtr_,t){} */
    
    
    /// Destructor
    ~FwdPtr() {}
    
    /// Dereference operator
    T const&
    operator*() const { return ptr_.isNonnull() ? ptr_.operator*() : backPtr_.operator*();}

    /// Member dereference operator
    T const*
    operator->() const{ return ptr_.isNonnull() ? ptr_.operator->() : backPtr_.operator->();}

    /// Returns C++ pointer to the item
    T const* get() const { return ptr_.isNonnull() ? ptr_.get() : backPtr_.get();}


    /// Checks for null
    bool isNull() const {return !isNonnull(); }
    
    /// Checks for non-null
    //bool isNonnull() const {return id().isValid(); }
    bool isNonnull() const {return ptr_.isNonnull() || backPtr_.isNonnull(); }
    /// Checks for null
    bool operator!() const {return isNull();}
    
    /// Checks if collection is in memory or available
    /// in the event. No type checking is done.
    bool isAvailable() const {return ptr_.isAvailable() || backPtr_.isAvailable();}
    
    /// Checks if this FwdPtr is transient (i.e. not persistable).
    bool isTransient() const {return ptr_.isTransient();}
    
    /// Accessor for product ID.
    ProductID id() const {return ptr_.isNonnull() ? ptr_.id() : backPtr_.id();}

    /// Accessor for product getter.
    EDProductGetter const* productGetter() const {
      if (ptr_.productGetter()) return ptr_.productGetter();
      else return backPtr_.productGetter();
    }

    key_type key() const {return ptr_.isNonnull() ? ptr_.key() : backPtr_.key() ;}
    
    bool hasProductCache() const { return ptr_.hasProductCache() || backPtr_.hasProductCache();} 
    
    RefCore const& refCore() const {return ptr_.isNonnull() ? ptr_.refCore() : backPtr_.refCore() ;}
    // ---------- member functions ---------------------------
    
    void const* product() const {return 0;}
    
    Ptr<value_type> const& ptr() const { return ptr_;}
    Ptr<value_type> const& backPtr() const { return backPtr_;}
      
    //Used by ROOT storage
    CMS_CLASS_VERSION(10)

  private:
    Ptr<value_type> ptr_;
    Ptr<value_type> backPtr_;
  };
  

  template<typename T>
  inline
  bool
  operator==(FwdPtr<T> const& lhs, FwdPtr<T> const& rhs) {
    return (lhs.ptr() == rhs.ptr() ||
	     lhs.backPtr() == rhs.ptr() ||
	     lhs.ptr() == rhs.backPtr() ||
	     lhs.backPtr() == rhs.backPtr());
  }

  template<typename T>
  inline
  bool
  operator!=(FwdPtr<T> const& lhs, FwdPtr<T> const& rhs) {
    return !(lhs == rhs);
  }

  template<typename T>
  inline
  bool
  operator<(FwdPtr<T> const& lhs, FwdPtr<T> const& rhs) {
    /// The ordering of integer keys guarantees that the ordering of FwdPtrs within
    /// a collection will be identical to the ordering of the referenced objects in the collection.
    return (lhs.ptr() < rhs.ptr());
  }

}
#endif
