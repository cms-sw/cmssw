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
// $Id: RefToBase.h,v 1.18 2007/05/30 09:17:06 llista Exp $
//

// system include files

// user include files

#include "boost/shared_ptr.hpp"

#include "Reflex/Object.h"
#include "Reflex/Type.h"

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/Utilities/interface/EDMException.h"


namespace edm {
  namespace reftobase {
    // The following makes ROOT::Reflex::Type available as reftobase::Type,
    // etc.
    using ROOT::Reflex::Type;
    using ROOT::Reflex::Object;

    //------------------------------------------------------------------
    // Class template BaseHolder<T>
    //
    // BaseHolder<T> is an abstract base class that manages a single
    // edm::Ref to an element of type T in a collection in the Event;
    // the purpose of this abstraction is to hide the type of the
    // collection from code that can not know about that type.
    // 
    //------------------------------------------------------------------
    template <class T>
    class BaseHolder {
    public:
      virtual ~BaseHolder();
      virtual BaseHolder<T>* clone() const = 0;

      // Return the address of the element to which the hidden Ref
      // refers.
      virtual T const* getPtr() const = 0;

      // Return the ProductID of the collection to which the hidden
      // Ref refers.
      virtual ProductID id() const = 0;

      // Check to see if the Ref hidden in 'rhs' is equal to the Ref
      // hidden in 'this'. They can not be equal if they are of
      // different types. Note that the equality test also returns
      // false if dynamic type of 'rhs' is different from the dynamic
      // type of 'this', *even when the hiddens Refs are actually
      // equivalent*.
      virtual bool isEqualTo(BaseHolder<T> const& rhs) const = 0;

      // If the type of Ref I contain matches the type contained in
      // 'fillme', set the Ref in 'fillme' equal to mine and return
      // true. If not, write the name of the type I really contain to
      // msg, and return false.
      virtual bool fillRefIfMyTypeMatches(RefHolderBase& fillme,
					  std::string& msg) const = 0;
    protected:
      // We want the following called only by derived classes.
      BaseHolder();
      BaseHolder(BaseHolder const& other);
      BaseHolder& operator= (BaseHolder const& rhs);

    private:
    };

    //------------------------------------------------------------------
    // Class template Holder<T,REF>
    //------------------------------------------------------------------

    template <class T, class REF>
    class Holder : public BaseHolder<T> {
    public:
      Holder();
      Holder(Holder const& other);
      explicit Holder(REF const& iRef);
      Holder& operator= (Holder const& rhs);
      void swap(Holder& other);
      virtual ~Holder();
      virtual BaseHolder<T>* clone() const;

      virtual T const* getPtr() const;
      virtual ProductID id() const;
      virtual bool isEqualTo(BaseHolder<T> const& rhs) const;
      REF const& getRef() const;

      virtual bool fillRefIfMyTypeMatches(RefHolderBase& fillme,
					  std::string& msg) const;

    private:
      REF ref_;
    };

    //------------------------------------------------------------------
    // Class template IndirectHolder<T>
    //------------------------------------------------------------------

    template <class T>
    class IndirectHolder : public BaseHolder<T> {
    public:
      // It may be better to use auto_ptr<RefHolderBase> in
      // this constructor, so that the cloning can be avoided. I'm not
      // sure if use of auto_ptr here causes any troubles elsewhere.
      IndirectHolder() { }
      IndirectHolder(boost::shared_ptr<RefHolderBase> p);
      IndirectHolder(IndirectHolder const& other);
      IndirectHolder& operator= (IndirectHolder const& rhs);
      void swap(IndirectHolder& other);
      virtual ~IndirectHolder();
      
      virtual BaseHolder<T>* clone() const;
      virtual T const* getPtr() const;
      virtual ProductID id() const;
      virtual bool isEqualTo(BaseHolder<T> const& rhs) const;

      virtual bool fillRefIfMyTypeMatches(RefHolderBase& fillme,
					  std::string& msg) const;

    private:
      RefHolderBase* helper_;
    };

    //------------------------------------------------------------------
    // Class RefHolderBase
    //------------------------------------------------------------------

    class RefHolderBase {
    public:
      template <class T> T const* getPtr() const;
      virtual ~RefHolderBase();
      virtual RefHolderBase* clone() const = 0;

      virtual ProductID id() const = 0;

      // Check to see if the Ref hidden in 'rhs' is equal to the Ref
      // hidden in 'this'. They can not be equal if they are of
      // different types.
      virtual bool isEqualTo(RefHolderBase const& rhs) const = 0;

      // If the type of Ref I contain matches the type contained in
      // 'fillme', set the Ref in 'fillme' equal to mine and return
      // true. If not, write the name of the type I really contain to
      // msg, and return false.

      virtual bool fillRefIfMyTypeMatches(RefHolderBase& ref,
					  std::string& msg) const = 0;

    private:
      // "cast" the real type of the element (the T of contained Ref),
      // and cast it to the type specified by toType, using Reflex.
      // Return 0 if the real type is not toType nor a subclass of
      // toType.
      virtual void const* pointerToType(Type const& toType) const = 0;
    };

    //------------------------------------------------------------------
    // Class template RefHolder<REF>
    //------------------------------------------------------------------

    template <class REF>
    class RefHolder : public RefHolderBase {
    public:
      RefHolder();
      explicit RefHolder(REF const& ref);
      RefHolder(RefHolder const& other);
      RefHolder& operator=(RefHolder const& rhs);
      void swap(RefHolder& other);
      virtual ~RefHolder();
      virtual RefHolderBase* clone() const;

      virtual ProductID id() const;
      virtual bool isEqualTo(RefHolderBase const& rhs) const;
      virtual bool fillRefIfMyTypeMatches(RefHolderBase& fillme,
					  std::string& msg) const;


      REF const& getRef() const;
      void setRef(REF const& r);

    private:
      virtual void const* pointerToType(Type const& iToType) const;
      REF ref_;
    };
    
    //------------------------------------------------------------------
    // Implementation of BaseHolder<T>
    //------------------------------------------------------------------

    template <class T>
    BaseHolder<T>::BaseHolder() 
    { }

    template <class T>
    BaseHolder<T>::BaseHolder(BaseHolder const& other)
    {
      // Nothing to do.
    }

    template <class T>
    BaseHolder<T>&
    BaseHolder<T>::operator= (BaseHolder<T> const& other)
    {
      // No data to assign.
      return *this;
    }

    template <class T>
    BaseHolder<T>::~BaseHolder()
    {
      // nothing to do.
    }

    //------------------------------------------------------------------
    // Implementation of Holder<T,REF>
    //------------------------------------------------------------------

    template <class T, class REF>
    inline
    Holder<T,REF>::Holder() : 
      ref_()
    {  }

    template <class T, class REF>
    inline
    Holder<T,REF>::Holder(Holder const& other) :
      ref_(other.ref_)
    { }

    template <class T, class REF>
    inline
    Holder<T,REF>::Holder(REF const& r) :
      ref_(r)
    { }

    template <class T, class REF>
    inline
    Holder<T,REF> &
    Holder<T,REF>::operator=(Holder const& rhs)
    {
      Holder temp(rhs);
      swap(temp);
      return *this;
    }

    template <class T, class REF>
    inline
    void
    Holder<T,REF>::swap(Holder& other)
    {
      std::swap(ref_, other.ref_);
    }

    template <class T, class REF>
    inline
    Holder<T,REF>::~Holder()
    { }

    template <class T, class REF>
    inline
    BaseHolder<T>*
    Holder<T,REF>::clone() const 
    {
      return new Holder(*this);
    }

    template <class T, class REF>
    inline
    T const*
    Holder<T,REF>::getPtr() const
    {
      return ref_.operator->();
    }

    template <class T, class REF>
    inline
    ProductID
    Holder<T,REF>::id() const
    {
      return ref_.id();
    }

    template <class T, class REF>
    inline
    bool
    Holder<T,REF>::isEqualTo(BaseHolder<T> const& rhs) const
    {
      Holder const* h = dynamic_cast<Holder const*>(&rhs);
      return h && (getRef() == h->getRef());
      //       if (h == 0) return false;
      //       return getRef() == h->getRef();
    }

    template <class T, class REF>
    inline
    REF const&
    Holder<T,REF>::getRef() const
    {
      return ref_;
    }

    template <class T, class REF>
    bool
    Holder<T,REF>::fillRefIfMyTypeMatches(RefHolderBase& fillme,
					  std::string& msg) const
    {
      RefHolder<REF>* h = dynamic_cast<RefHolder<REF>*>(&fillme);
      bool conversion_worked = (h != 0);

      if (conversion_worked)
 	h->setRef(ref_);
      else
	msg = typeid(REF).name();

      return conversion_worked;
    }

    //------------------------------------------------------------------
    // Implementation of RefHolderBase
    //------------------------------------------------------------------

    inline
    RefHolderBase::~RefHolderBase()
    { }

    template <class T>
    T const*
    RefHolderBase::getPtr() const
    {
      static Type s_type(Type::ByTypeInfo(typeid(T)));
      return static_cast<T const*>(pointerToType(s_type));
    }

    //------------------------------------------------------------------
    // Implementation of IndirectHolder<T>
    //------------------------------------------------------------------

    template <class T>
    inline
    IndirectHolder<T>::IndirectHolder(boost::shared_ptr<RefHolderBase> p) :
      helper_(p->clone()) 
    { }

    template <class T>
    inline
    IndirectHolder<T>::IndirectHolder(IndirectHolder const& other) : 
      helper_(other.helper_->clone()) 
    { }

    template <class T>
    inline
    IndirectHolder<T>& 
    IndirectHolder<T>::operator= (IndirectHolder const& rhs) 
    {
      IndirectHolder temp(rhs);
      swap(temp);
      return *this;
    }

    template <class T>
    inline
    void
    IndirectHolder<T>::swap(IndirectHolder& other) 
    {
      std::swap(helper_, other.helper_);
    }

    template <class T>
    IndirectHolder<T>::~IndirectHolder()
    {
      delete helper_;
    }

    template <class T>
    BaseHolder<T>* 
    IndirectHolder<T>::clone() const
    {
      return new IndirectHolder<T>(*this);
    }

    template <class T>
    T const* 
    IndirectHolder<T>::getPtr() const 
    {
      return helper_-> template getPtr<T>();
    }

    template <class T>
    ProductID
    IndirectHolder<T>::id() const
    {
      return helper_->id();
    }

    template <class T>
    bool
    IndirectHolder<T>::isEqualTo(BaseHolder<T> const& rhs) const 
    {
      IndirectHolder const* h = dynamic_cast<IndirectHolder const*>(&rhs);
      return h && helper_->isEqualTo(*h->helper_);
    }

    template <class T>
    bool
    IndirectHolder<T>::fillRefIfMyTypeMatches(RefHolderBase& fillme,
					      std::string& msg) const
    {
      return helper_->fillRefIfMyTypeMatches(fillme, msg);
    }


    //------------------------------------------------------------------
    // Implementation of RefHolder<REF>
    //------------------------------------------------------------------

    template <class REF>
    RefHolder<REF>::RefHolder() : 
      ref_()
    { }
  
    template <class REF>
    RefHolder<REF>::RefHolder(RefHolder const& rhs) :
      ref_( rhs.ref_ )
    { }

    template <class REF>
    RefHolder<REF>& RefHolder<REF>::operator=(RefHolder const& rhs) {
      ref_ = rhs.ref_; return *this;
    }

    template <class REF>
    RefHolder<REF>::RefHolder(REF const& ref) : 
      ref_(ref) 
    { }


    template <class REF>
    RefHolder<REF>::~RefHolder() 
    { }

    template <class REF>
    RefHolderBase* 
    RefHolder<REF>::clone() const
    {
      return new RefHolder(ref_);
    }

    template <class REF>
    ProductID
    RefHolder<REF>::id() const 
    {
      return ref_.id();
    }

    template <class REF>
    bool
    RefHolder<REF>::isEqualTo(RefHolderBase const& rhs) const 
    { 
      RefHolder const* h(dynamic_cast<RefHolder const*>(&rhs));
      return h && (getRef() == h->getRef());
    }

    template <class REF>
    bool 
    RefHolder<REF>::fillRefIfMyTypeMatches(RefHolderBase& fillme,
					   std::string& msg) const
    {
      RefHolder* h = dynamic_cast<RefHolder*>(&fillme);
      bool conversion_worked = (h != 0);
      if (conversion_worked)
	h->setRef(ref_);
      else
	msg = typeid(REF).name();
      return conversion_worked;
    }

    template <class REF>
    inline
    REF const&
    RefHolder<REF>::getRef() const
    {
      return ref_;
    }

    template <class REF>
    inline
    void
    RefHolder<REF>::swap(RefHolder& other)
    {
      std::swap(ref_, other.ref_);
    }

    template <class REF>
    inline
    void
    RefHolder<REF>::setRef(REF const& r)
    {
      ref_ = r;
    }

    template <class REF>
    void const* 
    RefHolder<REF>::pointerToType(Type const& iToType) const 
    {
      typedef typename REF::value_type contained_type;
      static const Type s_type(Type::ByTypeInfo(typeid(contained_type)));
    
      // The const_cast below is needed because
      // Object's constructor requires a pointer to
      // non-const void, although the implementation does not, of
      // course, modify the object to which the pointer points.
      Object obj(s_type, const_cast<void*>(static_cast<const void*>(ref_.get())));
      return obj.CastObject(iToType).Address(); // returns void*, after pointer adjustment
    }
  } // namespace reftobase

  //--------------------------------------------------------------------
  // Class template RefToBase<T>
  //--------------------------------------------------------------------

  /// RefToBase<T> provides a mechanism to refer to an object of type
  /// T (or which has T as a public base), held in a collection (of
  /// type not known to RefToBase<T>) which itself it in an Event.

  template <class T>
  class RefToBase
  {
  public:
    typedef T   value_type;

    RefToBase();
    RefToBase(RefToBase const& other);
    template <class REF> explicit RefToBase(REF const& r);
    RefToBase(boost::shared_ptr<reftobase::RefHolderBase> p);

    ~RefToBase();

    RefToBase const& operator= (RefToBase const& rhs);

    value_type const& operator*() const;
    value_type const* operator->() const;
    value_type const* get() const;

    ProductID id() const;

    template <class REF> REF castTo() const;

    bool isNull() const;
    bool isNonnull() const;
    bool operator!() const;

    bool operator==(RefToBase const& rhs) const;
    bool operator!=(RefToBase const& rhs) const;

    void swap(RefToBase& other);

  private:
    value_type const* getPtrImpl() const;
    reftobase::BaseHolder<value_type>* holder_;
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
  template <class REF>
  inline
  RefToBase<T>::RefToBase(REF const& iRef) : 
    holder_(new reftobase::Holder<T,REF>(iRef)) 
  { }

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
  RefToBase<T> const&
  RefToBase<T>:: operator= (RefToBase<T> const& iRHS) 
  {
    RefToBase<T> temp(iRHS);
    this->swap(temp);
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
    
  /// cast to a concrete type
  template <class T>
  template <class REF>
  REF
  RefToBase<T>::castTo() const
  {
    if (!holder_)
      {
	throw edm::Exception(errors::InvalidReference)
	  << "attempting to cast a null RefToBase;\n"
	  << "You should check for nullity before casting.";
      }

    reftobase::RefHolder<REF> concrete_holder;
    std::string hidden_ref_type;
    if (!holder_->fillRefIfMyTypeMatches(concrete_holder,
					 hidden_ref_type))
      {
	throw edm::Exception(errors::InvalidReference)
	  << "cast to type: " << typeid(REF).name()
	  << "\nfrom type: " << hidden_ref_type
	  << " failed. Catch this exception in case you need to check"
	  << " the concrete reference type.";
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
  T const*
  RefToBase<T>::getPtrImpl() const 
  {
    return holder_ ? holder_->getPtr() : 0;
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

#endif
