#ifndef DataFormats_GeometrySurface_ReferenceCounted_h
#define DataFormats_GeometrySurface_ReferenceCounted_h
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
//

// system include files
#include "boost/intrusive_ptr.hpp"
#include <atomic>

// user include files

// forward declarations

class BasicReferenceCounted {
public:
  BasicReferenceCounted() : referenceCount_(0) {}
  BasicReferenceCounted(const BasicReferenceCounted& /* iRHS */) : referenceCount_(0) {}
  BasicReferenceCounted(BasicReferenceCounted&&) : referenceCount_(0) {}
  BasicReferenceCounted& operator=(BasicReferenceCounted&&) { return *this; }

  BasicReferenceCounted& operator=(const BasicReferenceCounted&) { return *this; }

  virtual ~BasicReferenceCounted() {}

  // ---------- const member functions ---------------------

  void addReference() const { referenceCount_++; }
  void removeReference() const {
    if (1 == referenceCount_--) {
      delete const_cast<BasicReferenceCounted*>(this);
    }
  }

  unsigned int references() const { return referenceCount_; }
  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------

private:
  // ---------- member data --------------------------------
  mutable std::atomic<unsigned int> referenceCount_;
};

template <class T>
class ReferenceCountingPointer : public boost::intrusive_ptr<T> {
public:
  ReferenceCountingPointer(T* iT) : boost::intrusive_ptr<T>(iT) {}
  ReferenceCountingPointer() {}
};

template <class T>
class ConstReferenceCountingPointer : public boost::intrusive_ptr<const T> {
public:
  ConstReferenceCountingPointer(const T* iT) : boost::intrusive_ptr<const T>(iT) {}
  ConstReferenceCountingPointer() {}
  ConstReferenceCountingPointer(const ReferenceCountingPointer<T>& other) : boost::intrusive_ptr<const T>(&(*other)) {}
};

inline void intrusive_ptr_add_ref(const BasicReferenceCounted* iRef) { iRef->addReference(); }

inline void intrusive_ptr_release(const BasicReferenceCounted* iRef) { iRef->removeReference(); }

// condition uses naive RefCount
typedef BasicReferenceCounted ReferenceCountedInConditions;

typedef BasicReferenceCounted ReferenceCountedInEvent;

typedef BasicReferenceCounted ReferenceCounted;

#endif
