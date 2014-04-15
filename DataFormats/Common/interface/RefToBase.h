#ifndef DataFormats_Common_RefToBase_h
#define DataFormats_Common_RefToBase_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     RefToBase
//
/**\class RefToBase RefToBase.h DataFormats/Common/interface/RefToBase.h

Description: Interface to a reference to an item based on the base class of the item

Usage:
Using an edm:RefToBase<T> allows one to hold references to items in different containers
within the edm::Event where those objects are only related by a base class, T.

\code
edm::Ref<Foo> foo(...);
std::vector<edm::RefToBase<Bar> > bars;
bars.push_back(edm::RefToBase<Bar>(foo));
\endcode

Cast to concrete type can be done via the castTo<REF>
function template. This function throws an exception
if the type passed as REF does not match the concrete
reference type.

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Apr  3 16:37:59 EDT 2006
//

// system include files

// user include files

#include "boost/shared_ptr.hpp"
#include "boost/static_assert.hpp"
#include "boost/type_traits/is_base_of.hpp"

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Common/interface/BaseHolder.h"

#include "DataFormats/Common/interface/Holder.h"
#include "DataFormats/Common/interface/IndirectHolder.h"
#include "DataFormats/Common/interface/RefHolder.h"

namespace edm {
  //--------------------------------------------------------------------
  // Class template RefToBase<T>
  //--------------------------------------------------------------------

  /// RefToBase<T> provides a mechanism to refer to an object of type
  /// T (or which has T as a public base), held in a collection (of
  /// type not known to RefToBase<T>) which itself it in an Event.

  template<typename T> class RefToBaseVector;
  template<typename C, typename T, typename F> class Ref;
  template<typename C> class RefProd;
  template<typename T> class RefToBaseProd;
  template<typename T> class View;

  template <class T>
  class RefToBase
  {
  public:
    typedef T   value_type;

    RefToBase();
    RefToBase(RefToBase const& other);
    template <typename C1, typename T1, typename F1>
    explicit RefToBase(Ref<C1, T1, F1> const& r);
    template <typename C>
    explicit RefToBase(RefProd<C> const& r);
    RefToBase(RefToBaseProd<T> const& r, size_t i);
    RefToBase(Handle<View<T> > const& handle, size_t i);
    template <typename T1>
    explicit RefToBase(RefToBase<T1> const & r );
    RefToBase(boost::shared_ptr<reftobase::RefHolderBase> p);
    ~RefToBase();

    RefToBase& operator= (RefToBase const& rhs);

    value_type const& operator*() const;
    value_type const* operator->() const;
    value_type const* get() const;

    ProductID id() const;
    size_t key() const;

    template <class REF> REF castTo() const;

    bool isNull() const;
    bool isNonnull() const;
    bool operator!() const;

    bool operator==(RefToBase const& rhs) const;
    bool operator!=(RefToBase const& rhs) const;

    void swap(RefToBase& other);

    std::auto_ptr<reftobase::RefHolderBase> holder() const;

    EDProductGetter const* productGetter() const;
    bool hasProductCache() const;
    void const * product() const;

    /// Checks if collection is in memory or available
    /// in the Event. No type checking is done.
    bool isAvailable() const { return holder_? holder_->isAvailable(): false; }
    
    //Needed for ROOT storage
    CMS_CLASS_VERSION(10)
  private:
    value_type const* getPtrImpl() const;
    reftobase::BaseHolder<value_type>* holder_;
    friend class RefToBaseVector<T>;
    friend class RefToBaseProd<T>;
    template<typename B> friend class RefToBase;
  };

  //--------------------------------------------------------------------
  // Implementation of RefToBase<T>
  //--------------------------------------------------------------------

  template <class T>
  inline
  RefToBase<T>::RefToBase() :
    holder_(0)
  { }

  template <class T>
  inline
  RefToBase<T>::RefToBase(RefToBase const& other) :
    holder_(other.holder_  ? other.holder_->clone() : 0)
  { }

  template <class T>
  template <typename C1, typename T1, typename F1>
  inline
  RefToBase<T>::RefToBase(Ref<C1, T1, F1> const& iRef) :
    holder_(new reftobase::Holder<T,Ref<C1, T1, F1> >(iRef))
  { }

  template <class T>
  template <typename C>
  inline
  RefToBase<T>::RefToBase(RefProd<C> const& iRef) :
    holder_(new reftobase::Holder<T,RefProd<C> >(iRef))
  { }

  template <class T>
  template <typename T1>
  inline
  RefToBase<T>::RefToBase(RefToBase<T1> const& iRef) :
        holder_(new reftobase::IndirectHolder<T> (
             boost::shared_ptr< edm::reftobase::RefHolderBase>(iRef.holder().release())
        ) )
  {
    // OUT: holder_( new reftobase::Holder<T,RefToBase<T1> >(iRef ) )  {
    // Forcing the conversion through IndirectHolder,
    //   as Holder<T,RefToBase<T1>> would need dictionaries we will never have.
    // In this way we only need the IndirectHolder<T> and the RefHolder of the real type of the item
    // This might cause a small performance penalty.
    BOOST_STATIC_ASSERT( ( boost::is_base_of<T, T1>::value ) );
  }

  template <class T>
  inline
  RefToBase<T>::RefToBase(boost::shared_ptr<reftobase::RefHolderBase> p) :
    holder_(new reftobase::IndirectHolder<T>(p))
  { }

  template <class T>
  inline
  RefToBase<T>::~RefToBase()
  {
    delete holder_;
  }

  template <class T>
  inline
  RefToBase<T>&
  RefToBase<T>:: operator= (RefToBase<T> const& iRHS)
  {
    RefToBase<T> temp( iRHS);
    temp.swap(*this);
    return *this;
  }

  template <class T>
  inline
  T const&
  RefToBase<T>::operator*() const
  {
    return *getPtrImpl();
  }

  template <class T>
  inline
  T const*
  RefToBase<T>::operator->() const
  {
    return getPtrImpl();
  }

  template <class T>
  inline
  T const*
  RefToBase<T>::get() const
  {
    return getPtrImpl();
  }

  template <class T>
  inline
  ProductID
  RefToBase<T>::id() const
  {
    return  holder_ ? holder_->id() : ProductID();
  }

  template <class T>
  inline
  size_t
  RefToBase<T>::key() const
  {
    if ( holder_ == 0 )
	Exception::throwThis(errors::InvalidReference,
	  "attempting get key from  null RefToBase;\n"
	  "You should check for nullity before calling key().");
    return  holder_->key();
  }

  /// cast to a concrete type
  template <class T>
  template <class REF>
  REF
  RefToBase<T>::castTo() const
  {
    if (!holder_)
      {
	Exception::throwThis(errors::InvalidReference,
	  "attempting to cast a null RefToBase;\n"
	  "You should check for nullity before casting.");
      }

    reftobase::RefHolder<REF> concrete_holder;
    std::string hidden_ref_type;
    if (!holder_->fillRefIfMyTypeMatches(concrete_holder,
					 hidden_ref_type))
      {
	Exception::throwThis(errors::InvalidReference,
	  "cast to type: ",
	  typeid(REF).name(),
	  "\nfrom type: ",
	  hidden_ref_type.c_str(),
	  " failed. Catch this exception in case you need to check"
	  " the concrete reference type.");
      }
    return concrete_holder.getRef();
  }

  /// Checks for null
  template <class T>
  inline
  bool
  RefToBase<T>::isNull() const
  {
    return !id().isValid();
  }

  /// Checks for non-null
  template <class T>
  inline
  bool
  RefToBase<T>::isNonnull() const
  {
    return !isNull();
  }

  /// Checks for null
  template <class T>
  inline
  bool
  RefToBase<T>::operator!() const
  {
    return isNull();
  }

  template <class T>
  inline
  bool
  RefToBase<T>::operator==(RefToBase<T> const& rhs) const
  {
    return holder_
      ? holder_->isEqualTo(*rhs.holder_)
      : holder_ == rhs.holder_;
  }

  template <class T>
  inline
  bool
  RefToBase<T>::operator!=(RefToBase<T> const& rhs) const
  {
    return !(*this == rhs);
  }

  template <class T>
  inline
  void
  RefToBase<T>::swap(RefToBase<T> & other)
  {
    std::swap(holder_, other.holder_);
  }

  template <class T>
  inline
  EDProductGetter const* RefToBase<T>::productGetter() const {
    return holder_? holder_->productGetter():nullptr;
  }

  template <class T>
  inline
  bool RefToBase<T>::hasProductCache() const {
    return holder_?holder_->hasProductCache():false;
  }

  template <class T>
  inline
  void const * RefToBase<T>::product() const {
    return holder_?holder_->product():nullptr;
  }

  template <class T>
  inline
  T const*
  RefToBase<T>::getPtrImpl() const
  {
    return holder_ ? holder_->getPtr() : 0;
  }

  template <class T>
  std::auto_ptr<reftobase::RefHolderBase> RefToBase<T>::holder() const {
    return holder_? holder_->holder() : std::auto_ptr<reftobase::RefHolderBase>();
  }

  // Free swap function
  template <class T>
  inline
  void
  swap(RefToBase<T>& a, RefToBase<T>& b)
  {
    a.swap(b);
  }
}

#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"

namespace edm {
  template <class T>
  inline
  RefToBase<T>::RefToBase(RefToBaseProd<T> const& r, size_t i) :
    holder_( r.operator->()->refAt( i ).holder_->clone() ) {
  }

  template<typename T>
  inline
  RefToBase<T>::RefToBase(Handle<View<T> > const& handle, size_t i) :
    holder_( handle.operator->()->refAt( i ).holder_->clone() ) {
  }

}

#endif
