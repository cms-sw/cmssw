#ifndef FWCore_Utilities_atomic_value_ptr_h
#define FWCore_Utilities_atomic_value_ptr_h

// ----------------------------------------------------------------------
//
// atomic_value_ptr.h - Smart pointer permitting copying of pointed-to object.
//
// This is an atomic version of value_ptr.
//
// The prologue of FWCore/Utilities/interface/value_ptr.h
// describes the functionality of value_ptr.
//
// This allows the value of the pointer to be changed atomically.
// Note that copy/move construction amd copy/move assignment
// are *not* atomic, as an object of type T must be copied or moved.
// ----------------------------------------------------------------------

#ifndef __GCCXML__
#include <atomic>
#endif

#include <memory>

#ifdef __GCCXML__
#define nullptr 0
#endif

namespace edm {

  // --------------------------------------------------------------------
  //
  //  Auxiliary traits class template providing default clone()
  //  Users should specialize this template for types that have their
  //  own self-copy operations; failure to do so may lead to slicing!
  //  User specified clone methods must be thread safe.
  //
  // --------------------------------------------------------------------


  template <typename T>
  struct atomic_value_ptr_traits {
    static T* clone(T const* p) { return new T(*p); }
  };

  // --------------------------------------------------------------------
  //
  // Copyable smart pointer class template using an atomic pointer.
  //
  // --------------------------------------------------------------------


  template <typename T>
  class atomic_value_ptr {

  public:

    // --------------------------------------------------
    // Default constructor/destructor:
    // --------------------------------------------------

    atomic_value_ptr() : myP(nullptr) { }
    explicit atomic_value_ptr(T* p) : myP(p) { }
    ~atomic_value_ptr() { delete myP.load(); }

    // --------------------------------------------------
    // Copy constructor/copy assignment:
    // --------------------------------------------------

    atomic_value_ptr(atomic_value_ptr const& orig) :
      myP(createFrom(orig.myP.load())) {
    }

    atomic_value_ptr& operator=(atomic_value_ptr const& orig) {
      atomic_value_ptr<T> local(orig);
      exchangeWithLocal(local);
      return *this;
    }

#ifndef __GCCXML__
    atomic_value_ptr(atomic_value_ptr&& orig) :
      myP(orig.myP) { orig.myP.store(nullptr); }

    atomic_value_ptr& operator=(atomic_value_ptr&& orig) {
      atomic_value_ptr<T> local(orig);
      exchangeWithLocal(local);
      return *this;
    }
#endif

    // --------------------------------------------------
    // Access mechanisms:
    // --------------------------------------------------

    T& operator*() const { return *myP; }
    T* operator->() const { return myP.load(); }

    // --------------------------------------------------
    // Copy-like construct/assign from compatible atomic_value_ptr<>:
    // --------------------------------------------------

    template <typename U>
    atomic_value_ptr(atomic_value_ptr<U> const& orig) :
      myP(createFrom(orig.operator->())) {
    }

    template <typename U>
    atomic_value_ptr& operator=(atomic_value_ptr<U> const& orig) {
      atomic_value_ptr<T> local(orig);
      exchangeWithLocal(local);
      return *this;
    }

    // --------------------------------------------------
    // Copy-like construct/assign from auto_ptr<>:
    // --------------------------------------------------

    atomic_value_ptr(std::auto_ptr<T> orig) :
      myP(orig.release()) {
    }

    atomic_value_ptr& operator=(std::auto_ptr<T> orig) {
      atomic_value_ptr<T> local(orig);
      exchangeWithLocal(local);
      return *this;
    }

#ifndef __GCCXML__
    // --------------------------------------------------
    // move-like construct/assign from unique_ptr<>:
    // --------------------------------------------------

    atomic_value_ptr(std::unique_ptr<T> orig) :
      myP(orig.release()) {
      orig = nullptr;
    }

    atomic_value_ptr& operator=(std::unique_ptr<T> orig) {
      atomic_value_ptr<T> local(orig);
      exchangeWithLocal(local);
      return *this;
    }
#endif

  T* load() const {
    return myP.load();
  }

  bool compare_exchange_strong(T*& oldValue, T* newValue) {
   return myP.compare_exchange_strong(oldValue, newValue); 
  }

  // The following typedef, function, and operator definition
  // support the following syntax:
  //   atomic_value_ptr<T> ptr(..);
  //   if (ptr) { ...
  // Where the conditional will evaluate as true if and only if the
  // pointer atomic_value_ptr contains is not null.
  private:
    typedef void (atomic_value_ptr::*bool_type)() const;
    void this_type_does_not_support_comparisons() const {}

  public:
    operator bool_type() const {
      return myP != nullptr ?
        &atomic_value_ptr<T>::this_type_does_not_support_comparisons : nullptr;
    }

  private:

    // --------------------------------------------------
    // Manipulation:
    // --------------------------------------------------

    // This function must be invoked by the object available across threads.
    // The argument "local" must reference an object local to one thread.
    void exchangeWithLocal(atomic_value_ptr& local) {
      T* old = myP.exchange(local.myP.load());
      local.myP = old;
    }

    // --------------------------------------------------
    // Implementation aid:
    // --------------------------------------------------

    template <typename U>
    static T*
    createFrom(U const* p) {
      return p
	? atomic_value_ptr_traits<U>::clone(p)
	: nullptr;
    }

    // --------------------------------------------------
    // Member data:
    // --------------------------------------------------

#ifndef __GCCXML__
    mutable std::atomic<T*> myP;
#else
    T* myP;
#endif

  }; // atomic_value_ptr

  // Do not allow nonsensical comparisons that the bool_type
  // conversion operator definition above would otherwise allow.
  // The function call inside the next 4 operator definitions is
  // private, so compilation will fail if there is an atempt to
  // instantiate these 4 operators.
  template <typename T, typename U>
  inline bool operator==(atomic_value_ptr<T> const& lhs, U const& rhs) {
    lhs.this_type_does_not_support_comparisons();	
    return false;	
  }

  template <typename T, typename U>
  inline bool operator!=(atomic_value_ptr<T> const& lhs, U const& rhs) {
    lhs.this_type_does_not_support_comparisons();	
    return false;	
  }

  template <typename T, typename U>
  inline bool operator==(U const& lhs, atomic_value_ptr<T> const& rhs) {
    rhs.this_type_does_not_support_comparisons();	
    return false;	
  }

  template <typename T, typename U>
  inline bool operator!=(U const& lhs, atomic_value_ptr<T> const& rhs) {
    rhs.this_type_does_not_support_comparisons();	
    return false;	
  }
}


#endif // FWCoreUtilities_atomic_value_ptr_h
