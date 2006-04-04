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
    Using an edm::OwnVector< RefToBase<T> > allows one to hold references to items in different containers
within the edm::Event where those objects are only related by a base class, T.

    The easiest way to create the RefToBase<> is to use the convenience function makeRefToBase()
/code
   edm::Ref<Foo> foo(...);
   edm::OwnVector<Bar> bars;
   bars.push_back( makeRefToBase<Bar>( foo ).release() );
/endcode

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Apr  3 16:37:59 EDT 2006
// $Id$
//

// system include files

// user include files

// forward declarations
namespace edm {
  template <class T>
  class RefToBase
{

   public:
      RefToBase() {}
      virtual ~RefToBase() {}

      // ---------- const member functions ---------------------
      virtual RefToBase* clone() const = 0;

      const T& operator*() const { return *getPtrImpl(); }
      const T* operator->() const { return getPtrImpl(); }
      const T* get() const { return getPtrImpl();}
      
      // ---------- member functions ---------------------------

   private:
      //RefToBase(const RefToBase&); // stop default

      //const RefToBase& operator=(const RefToBase&); // stop default

      // ---------- member data --------------------------------
      ///inheriting classes implement this to give access to referred object  
      virtual const T* getPtrImpl() const = 0;
      
};

  template<class T, class TRef>
  class RefToBaseImpl : public RefToBase<T> 
{
public:
  RefToBaseImpl() {}
  RefToBaseImpl(const TRef& iRef) : ref_(iRef) {}
  
  virtual RefToBase<T>* clone() const { return new RefToBaseImpl<T,TRef>(*this); }
private:
  virtual const T* getPtrImpl() const { return ref_.operator->(); }
  TRef ref_;
};

template< class T, class TRef>
std::auto_ptr<RefToBase<T> > makeRefToBase(const TRef& iRef) {
  return std::auto_ptr<RefToBase<T> >(new RefToBaseImpl<T,TRef>(iRef));
}
}
#endif
