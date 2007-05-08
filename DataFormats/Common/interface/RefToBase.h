#ifndef Common_RefToBase_h
#define Common_RefToBase_h
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

Cast to concrete type can be done via the castTo<TRef> 
function template. This function throws an exception
if the type passed as TRef does not match the concrete
reference type.

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Apr  3 16:37:59 EDT 2006
// $Id: RefToBase.h,v 1.13 2007/03/29 22:56:31 wmtan Exp $
//

// system include files
#include <algorithm>

// user include files

#include "Reflex/Object.h"
#include "Reflex/Type.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/EDMException.h"


namespace edm {
  namespace reftobase {
    template <class T>
    class BaseHolder {
    public:
      BaseHolder() {}
      virtual ~BaseHolder() {}
      virtual BaseHolder<T>* clone() const = 0;
      virtual const T* getPtr() const = 0;
      virtual ProductID id() const = 0;
      virtual bool isEqualTo(const BaseHolder<T> & rhs) const = 0;
    };

    template <class T, class TRef>
    class Holder : public BaseHolder<T> {
    public:
      Holder() {}
      explicit Holder(const TRef& iRef) : ref_(iRef) {}
      virtual ~Holder() {}
      virtual BaseHolder<T>* clone() const { return new Holder<T,TRef>(*this); }
      virtual const T* getPtr() const { return ref_.operator->(); }
      virtual ProductID id() const { return ref_.id(); }
      virtual bool isEqualTo(const BaseHolder<T> & rhs) const {
	const Holder<T, TRef> * h = dynamic_cast<const Holder<T, TRef> *>(& rhs);
	if (h == 0) return false;
	return getRef() == h->getRef();
      }
      const TRef & getRef() const { return ref_; }
    private:
      TRef ref_;
    };

    class IndirectHolderBaseHelper {
    public:
      template <class T>
      T const*
      getPtr() const {
	static ROOT::Reflex::Type s_type(ROOT::Reflex::Type::ByTypeInfo(typeid(T)));
	return static_cast<T const*>(pointerToType(s_type));
      }

      virtual ProductID id() const = 0;
      virtual bool isEqualTo(IndirectHolderBaseHelper const& rhs) const = 0;

    private:
      // "cast" the real type into toType, using Reflex
      virtual void const* pointerToType(ROOT::Reflex::Type const& toType) const = 0;
    };

    template <class T>
    class IndirectHolder : public BaseHolder<T> {
    public:
      IndirectHolder(IndirectHolder<T> const& rhs) : helper_(rhs.helper_->clone()) { }
      virtual BaseHolder<T>* clone() const { return new IndirectHolder<T>(*this); }
      virtual T const* getPtr() const { return helper_-> template getPtr<T>(); }
      virtual ProductID id() const { return helper_->id(); }

      // Chris wanted the following, but it can't be had because the
      // signature is wrong. The base class's function takes a
      // reference to the base class as an argument. We can't change
      // the signature without creating an overload, rather than an
      // override of the virtual function.

      virtual bool isEqualTo(BaseHolder<T> const& rhs) const {
	IndirectHolder const* h = dynamic_cast<IndirectHolder const*>(&rhs);
	return h ? helper_->isEqualTo(h->helper_) : false;
      }

    private:
      IndirectHolderBaseHelper* helper_;
    };

    template <class REF>
    class IndirectHolderHelper : public IndirectHolderBaseHelper {
    public:
      virtual ProductID id() const { return ref_.id(); }

      virtual bool isEqualTo(IndirectHolderBaseHelper const& rhs) const { 
	// Check to see if we're really the same type, and if so, are we really equal.
	return false; 
      }

    private:
      void const* pointerToType(ROOT::Reflex::Type const& iToType) const {
	typedef typename REF::value_type contained_type;
	static ROOT::Reflex::Type s_type(ROOT::Reflex::Type::ByTypeInfo(typeid(contained_type)));
	ROOT::Reflex::Object obj(s_type, ref_->get());
	return obj.CastObject(iToType).Address(); // returns void*, after pointer adjustment
      }
      REF ref_;
    };
  }

  
  template <class T>
  class RefToBase {
  public:
    RefToBase() : holder_(0) { }


    template <class TRef>
    explicit RefToBase(const TRef& iRef) : holder_(new reftobase::Holder<T,TRef>(iRef)) { }

    RefToBase(const RefToBase<T>& iOther): 
      holder_((0==iOther.holder_) ? static_cast<reftobase::BaseHolder<T>*>(0) : iOther.holder_->clone()) {
      }
    const RefToBase& operator=(const RefToBase<T>& iRHS) {
      RefToBase<T> temp(iRHS);
      this->swap(temp);
      return *this;
    }
    ~RefToBase() { delete holder_; }
    
    // ---------- const member functions ---------------------
    
    const T& operator*() const { return *getPtrImpl(); }
    const T* operator->() const { return getPtrImpl(); }
    const T* get() const { return getPtrImpl();}
    
    /// Accessor for product ID.
    ProductID id() const { 
      return  0 == holder_ ? ProductID() : holder_->id();
    }
    
    /// cast to a concrete type
    template<typename TRef>
    TRef castTo() const {
      typedef reftobase::Holder<T,TRef> Holder;
      const Holder * h = dynamic_cast<Holder *>(holder_);
      if (h == 0) {
	throw edm::Exception(errors::InvalidReference) 
	  << "trying to cast a RefToBase to the wrong type."
	  << "Catch this exception in case you need to check"
	  <<"the concrete reference type.";
      }
      return h->getRef();
    }
    
    /// Checks for null
    bool isNull() const { return !id().isValid(); }
    
    /// Checks for non-null
    bool isNonnull() const { return !isNull(); }
    
    /// Checks for null
    bool operator!() const { return isNull(); }
    
    bool operator==(const RefToBase<T> & rhs) const {
      return holder_->isEqualTo(*rhs.holder_);
    }
    
    bool operator!=(const RefToBase<T> & rhs) const {
      return !(*this == rhs);
    }
    
    // ---------- member functions ---------------------------
    void swap(RefToBase<T> & iOther) {
      std::swap(holder_, iOther.holder_);
    }
    
  private:
    
    // ---------- member data --------------------------------
    const T* getPtrImpl() const {
      if(0 == holder_) { return 0;}
      return holder_->getPtr();
    }
    
    reftobase::BaseHolder<T>* holder_;
  };
  
  // Free swap function
  template <class T>
  inline
  void
  swap(RefToBase<T>& a, RefToBase<T>& b) {
    a.swap(b);
  }

}

#endif
