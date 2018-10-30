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
edm::Ref<FooCollection> foo(...);
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

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Common/interface/BaseHolder.h"

#include "DataFormats/Common/interface/Holder.h"
#include "DataFormats/Common/interface/IndirectHolder.h"
#include "DataFormats/Common/interface/RefHolder.h"

#include <memory>
#include <type_traits>

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
    RefToBase(RefToBase && other) noexcept;
    RefToBase & operator=(RefToBase && other) noexcept;

    template <typename C1, typename T1, typename F1>
    explicit RefToBase(Ref<C1, T1, F1> const& r);
    template <typename C>
    explicit RefToBase(RefProd<C> const& r);
    RefToBase(RefToBaseProd<T> const& r, size_t i);
    RefToBase(Handle<View<T> > const& handle, size_t i);
    template <typename T1>
    explicit RefToBase(RefToBase<T1> const & r );
    RefToBase(std::unique_ptr<reftobase::BaseHolder<value_type>>);
    RefToBase(std::shared_ptr<reftobase::RefHolderBase> p);

    ~RefToBase() noexcept;

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

    std::unique_ptr<reftobase::RefHolderBase> holder() const;

    EDProductGetter const* productGetter() const;

    /// Checks if collection is in memory or available
    /// in the Event. No type checking is done.
    bool isAvailable() const { return holder_? holder_->isAvailable(): false; }

    bool isTransient() const { return holder_ ? holder_->isTransient() : false; }

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
    holder_(nullptr)
  { }

  template <class T>
  inline
  RefToBase<T>::RefToBase(RefToBase const& other) :
    holder_(other.holder_  ? other.holder_->clone() : nullptr)
  { }

  template <class T>
  inline
  RefToBase<T>::RefToBase(RefToBase && other) noexcept :
    holder_(other.holder_) { other.holder_=nullptr;}

  template <class T>
  inline
  RefToBase<T>& RefToBase<T>::operator=(RefToBase && other) noexcept {
    delete holder_; holder_=other.holder_; other.holder_=nullptr; return *this;
  }


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
             std::shared_ptr< edm::reftobase::RefHolderBase>(iRef.holder().release())
        ) )
  {
    // OUT: holder_( new reftobase::Holder<T,RefToBase<T1> >(iRef ) )  {
    // Forcing the conversion through IndirectHolder,
    //   as Holder<T,RefToBase<T1>> would need dictionaries we will never have.
    // In this way we only need the IndirectHolder<T> and the RefHolder of the real type of the item
    // This might cause a small performance penalty.
    static_assert( std::is_base_of<T, T1>::value, "RefToBase::RefToBase T not base of T1" );
  }

  template <class T>
  inline
  RefToBase<T>::RefToBase(std::unique_ptr<reftobase::BaseHolder<value_type>> p):
    holder_(p.release())
  {}

  template <class T>
  inline
  RefToBase<T>::RefToBase(std::shared_ptr<reftobase::RefHolderBase> p) :
    holder_(new reftobase::IndirectHolder<T>(p))
  { }

  template <class T>
  inline
  RefToBase<T>::~RefToBase() noexcept
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
    if ( holder_ == nullptr )
    {
        Exception::throwThis(errors::InvalidReference,
          "attempting get key from  null RefToBase;\n"
          "You should check for nullity before calling key().");
        return 0;
    }
    return  holder_->key();
  }

  namespace {
    // If the template parameters are classes or structs they should be
    // related by inheritance, otherwise they should be the same type.
    template<typename T, typename U>
    typename std::enable_if<std::is_class<T>::value>::type
    checkTypeCompatibility() { static_assert(std::is_base_of<T, U>::value ||
                                             std::is_base_of<U, T>::value,
                                             "RefToBase::castTo error element types are not related by inheritance"); }

    template<typename T, typename U>
    typename std::enable_if<!std::is_class<T>::value>::type
    checkTypeCompatibility() { static_assert(std::is_same<T, U>::value,
                               "RefToBase::castTo error non-class element types are not the same"); }

    // Convert the pointer types, use dynamic_cast if they are classes
    template<typename T, typename OUT>
    typename std::enable_if<std::is_class<T>::value, OUT const*>::type
    convertTo(T const* t) { return dynamic_cast<OUT const*>(t); }

    template<typename T, typename OUT>
    typename std::enable_if<!std::is_class<T>::value, OUT const*>::type
    convertTo(T const* t) { return t;}
  }

  template <class T>
  template <class REF>
  REF
  RefToBase<T>::castTo() const {

    if (!holder_) {
      Exception::throwThis(errors::InvalidReference,
                           "attempting to cast a null RefToBase;\n"
                           "You should check for nullity before casting.");
    }

    checkTypeCompatibility<T, typename REF::value_type>();

    // If REF is type edm::Ref<C,T,F>, then it is impossible to
    // check the container type C here. We just have to assume
    // that the caller provided the correct type.

    EDProductGetter const* getter = productGetter();
    if(getter) {
      return REF(id(), key(), getter);
    }

    T const* value = get();
    if(value == nullptr) {
      return REF(id());
    }
    typename REF::value_type const* newValue = convertTo<T, typename REF::value_type>(value);
    if(newValue) {
      return REF(id(), newValue, key(), isTransient());
    }

    Exception::throwThis(errors::InvalidReference,
                         "RefToBase<T>::castTo Error attempting to cast mismatched types\n"
                         "casting from RefToBase with T: ",
                         typeid(T).name(),
                         "\ncasting to: ",
                         typeid(REF).name()
                         );
    return REF();
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
  T const*
  RefToBase<T>::getPtrImpl() const
  {
    return holder_ ? holder_->getPtr() : nullptr;
  }

  template <class T>
  std::unique_ptr<reftobase::RefHolderBase> RefToBase<T>::holder() const {
    return holder_? holder_->holder() : std::unique_ptr<reftobase::RefHolderBase>();
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
