#ifndef SURFACE_REFERENCECOUNTED_H
#define SURFACE_REFERENCECOUNTED_H
// -*- C++ -*-
//
// Package:     Surface
// Class  :     ReferenceCounted
// 
/**\class ReferenceCounted ReferenceCounted.h DataFormats/GeometrySurface/interface/ReferenceCounted.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jul 15 09:17:20 EDT 2005
// $Id: ReferenceCounted.h,v 1.12 2013/01/20 08:09:02 innocent Exp $
//

// system include files
#include "boost/intrusive_ptr.hpp"

// user include files

// forward declarations

class BasicReferenceCounted
{

   public:
      BasicReferenceCounted() : referenceCount_(0) {}
      BasicReferenceCounted( const BasicReferenceCounted& iRHS ) : referenceCount_(0) {}

      const BasicReferenceCounted& operator=( const BasicReferenceCounted& ) {
	return *this;
      }
      virtual ~BasicReferenceCounted() {}

      // ---------- const member functions ---------------------

      void addReference() const { ++referenceCount_ ; }
      void removeReference() const { if( 0 == --referenceCount_ ) {
	  delete const_cast<BasicReferenceCounted*>(this);
	}
      }

      unsigned int  references() const {return referenceCount_;}

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:

      // ---------- member data --------------------------------
      mutable unsigned int referenceCount_;
};

template <class T> class ReferenceCountingPointer : 
  public boost::intrusive_ptr<T> 
{
 public:
  ReferenceCountingPointer(T* iT) : boost::intrusive_ptr<T>(iT) {}
  ReferenceCountingPointer() {}
};

template <class T> class ConstReferenceCountingPointer : 
  public boost::intrusive_ptr<const T> 
{
 public:
  ConstReferenceCountingPointer(const T* iT) : boost::intrusive_ptr<const T>(iT) {}
  ConstReferenceCountingPointer() {}
  ConstReferenceCountingPointer( const ReferenceCountingPointer<T>& other) :
    boost::intrusive_ptr<const T>(&(*other)) {}
};

inline void intrusive_ptr_add_ref( const BasicReferenceCounted* iRef ) {
  iRef->addReference();
}

inline void intrusive_ptr_release( const BasicReferenceCounted* iRef ) {
  iRef->removeReference();
}


// does not increase ref count
// delete if count is 0 (takes owership of orphans )
// clone if count is 0 
// double delete if deleted after the real owner...
template<typename T>
class MixedReference {
  explicit  MixedReference(T const * ip=nullptr) noexcept : p(ip){}
  ~MixedReference() noexcept { destroy();}

#ifndef CMS_NOCXX11
    MixedReference( MixedReference&& rh) noexcept : p(rh.p){rh.p=0;}
  MixedReference& operator=( MixedReference&& rh) noexcept {
    destroy();
    p=rh.p; rh.p=0;
    return *this;
}
#endif
  
  MixedReference( MixedReference const & rh) : p(rh.p? rh.p->clone(): nullptr ){}
  MixedReference& operator=( MixedReference const & rh) { 
    if (rh.p!=p) {
      destroy();
      p = rh.p? rh.p->clone(): nullptr;
    }
    return * this;
  }


  T const * get() const {return p;}
  T const & operator*() const { return *p;}

private:
    void destroy() noexcept {
      if (p && p->references()==0) { delete const_cast<T*>(p); }
    }
  T const * p;
};


#define CMSSW_POOLALLOCATOR

#ifdef CMSSW_POOLALLOCATOR
#include "DataFormats/GeometrySurface/interface/BlockWipedAllocator.h"
#else
template<typename T>
struct LocalCache {
  std::auto_ptr<T> ptr;
};

#endif

class ReferenceCountedPoolAllocated
#ifdef CMSSW_POOLALLOCATOR
  : public BlockWipedPoolAllocated
#endif
{
      
public:
  static int s_alive;
  static int s_referenced;

  ReferenceCountedPoolAllocated() : referenceCount_(0) { 
    s_alive++;
  }

  ReferenceCountedPoolAllocated( const ReferenceCountedPoolAllocated& iRHS ) : referenceCount_(0) {
    s_alive++;
  }
  
  const ReferenceCountedPoolAllocated& operator=( const ReferenceCountedPoolAllocated& ) {
    return *this;
  }

  virtual ~ReferenceCountedPoolAllocated() {
    s_alive--;
  }
  
  // ---------- const member functions ---------------------
  
  void addReference() const { ++referenceCount_ ; s_referenced++; }
  void removeReference() const { 
    s_referenced--;
    if( 0 == --referenceCount_ ) {
      delete const_cast<ReferenceCountedPoolAllocated*>(this);
    }
  }
  
  unsigned int  references() const {return referenceCount_;}
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  
   private:
  
  // ---------- member data --------------------------------
  mutable unsigned int referenceCount_;
};

inline void intrusive_ptr_add_ref( const ReferenceCountedPoolAllocated* iRef ) {
  iRef->addReference();
}

inline void intrusive_ptr_release( const ReferenceCountedPoolAllocated* iRef ) {
  iRef->removeReference();
} 

// condition uses naive RefCount
typedef BasicReferenceCounted ReferenceCountedInConditions;


// transient objects in algo and events are "poo allocated"
typedef ReferenceCountedPoolAllocated  ReferenceCountedInEvent;

// just to avoid changing all around 
// typedef ReferenceCountedPoolAllocated ReferenceCounted;
typedef BasicReferenceCounted ReferenceCounted;



#endif /* SURFACE_REFERENCECOUNTED_H */
