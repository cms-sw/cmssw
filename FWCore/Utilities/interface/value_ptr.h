#ifndef FWCore_Utilities_value_ptr_h
#define FWCore_Utilities_value_ptr_h

// ----------------------------------------------------------------------
//
// value_ptr.h - Smart pointer permitting copying of pointed-to object.
//
// Purpose: value_ptr provides a smart pointer template that provides
// sole ownership of the pointed-to object. When a value_ptr object is
// copied, the new value_ptr object is given a copy of the object pointed
// to by the original value_ptr object.
//
// The value_ptr_traits template is provided to allow specialization
// of the copying behavior. See the notes below.
//
// Use value_ptr only when deep-copying of the pointed-to object is
// desireable. Use boost::shared_ptr or std::shared_ptr when sharing
// of the pointed-to  object is desirable. Use std::unique_ptr
// when no copying is desirable.
//
// The design of value_ptr is taken from Herb Sutter's More
// Exceptional C++, with modifications by Marc Paterno and further
// modifications by Walter Brown. This version is based on the
// ValuePtr found in the Fermilab ZOOM library.
//
//
// Supports the following syntax
//   value_ptr<T> ptr(...);
//   if (ptr) { ...
// Where the conditional will evaluate as true if and only if the
// pointer the value_ptr contains is not null.
//
// ----------------------------------------------------------------------

#include <algorithm> // for std::swap()
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
  //
  // --------------------------------------------------------------------


  template <typename T>
  struct value_ptr_traits {
    static T* clone(T const* p) { return new T(*p); }
  };

  // --------------------------------------------------------------------
  //
  // Copyable smart pointer class template
  //
  // --------------------------------------------------------------------


  template <typename T>
  class value_ptr {

  public:

    // --------------------------------------------------
    // Default constructor/destructor:
    // --------------------------------------------------

    value_ptr() : myP(nullptr) { }
    explicit value_ptr(T* p) : myP(p) { }
    ~value_ptr() { delete myP; }

    // --------------------------------------------------
    // Copy constructor/copy assignment:
    // --------------------------------------------------

    value_ptr(value_ptr const& orig) :
      myP(createFrom(orig.myP)) {
    }

    value_ptr& operator=(value_ptr const& orig) {
      value_ptr<T> temp(orig);
      swap(temp);
      return *this;
    }

#ifndef __GCCXML__
    // --------------------------------------------------
    // Move constructor/move assignment:
    // --------------------------------------------------

    value_ptr(value_ptr&& orig) :
      myP(orig.myP) { orig.myP=nullptr; }

    value_ptr& operator=(value_ptr&& orig) {
      if (myP!=orig.myP) {
        delete myP;
        myP=orig.myP;
        orig.myP=nullptr;
      } 
      return *this;
    }
#endif

    // --------------------------------------------------
    // Access mechanisms:
    // --------------------------------------------------

    T& operator*() const { return *myP; }
    T* operator->() const { return myP; }

    // --------------------------------------------------
    // Manipulation:
    // --------------------------------------------------

    void swap(value_ptr& orig) { std::swap(myP, orig.myP); }

    // --------------------------------------------------
    // Copy-like construct/assign from compatible value_ptr<>:
    // --------------------------------------------------

    template <typename U>
    value_ptr(value_ptr<U> const& orig) :
      myP(createFrom(orig.operator->())) {
    }

    template <typename U>
    value_ptr& operator=(value_ptr<U> const& orig) {
      value_ptr<T> temp(orig);
      swap(temp);
      return *this;
    }

    // --------------------------------------------------
    // Copy-like construct/assign from auto_ptr<>:
    // --------------------------------------------------

    value_ptr(std::auto_ptr<T> orig) :
      myP(orig.release()) {
    }

    value_ptr& operator=(std::auto_ptr<T> orig) {
      value_ptr<T> temp(orig);
      swap(temp);
      return *this;
    }

#ifndef __GCCXML__
    // --------------------------------------------------
    // Move-like construct/assign from unique_ptr<>:
    // --------------------------------------------------

    value_ptr(std::unique_ptr<T> orig) :
      myP(orig.release()) { orig=nullptr; }

    value_ptr& operator=(std::unique_ptr<T> orig) {
      value_ptr<T> temp(orig);
      swap(temp);
      return *this;
    }
#endif

  // The following typedef, function, and operator definition
  // support the following syntax:
  //   value_ptr<T> ptr(..);
  //   if (ptr) { ...
  // Where the conditional will evaluate as true if and only if the
  // pointer value_ptr contains is not null.
  private:
    typedef void (value_ptr::*bool_type)() const;
    void this_type_does_not_support_comparisons() const {}

  public:
    operator bool_type() const {
      return myP != nullptr ?
        &value_ptr<T>::this_type_does_not_support_comparisons : nullptr;
    }

  private:

    // --------------------------------------------------
    // Implementation aid:
    // --------------------------------------------------

    template <typename U>
    static T*
    createFrom(U const* p) {
      return p
	? value_ptr_traits<U>::clone(p)
	: nullptr;
    }

    // --------------------------------------------------
    // Member data:
    // --------------------------------------------------

    T* myP;

  }; // value_ptr


  // --------------------------------------------------------------------
  //
  // Free-standing swap()
  //
  // --------------------------------------------------------------------

  template <typename T>
  inline
  void
  swap(value_ptr<T>& vp1, value_ptr<T>& vp2) { vp1.swap(vp2); }

  // Do not allow nonsensical comparisons that the bool_type
  // conversion operator definition above would otherwise allow.
  // The function call inside the next 4 operator definitions is
  // private, so compilation will fail if there is an attempt to
  // instantiate these 4 operators.
  template <typename T, typename U>
  inline bool operator==(value_ptr<T> const& lhs, U const& rhs) {
    lhs.this_type_does_not_support_comparisons();	
    return false;	
  }

  template <typename T, typename U>
  inline bool operator!=(value_ptr<T> const& lhs, U const& rhs) {
    lhs.this_type_does_not_support_comparisons();	
    return false;	
  }

  template <typename T, typename U>
  inline bool operator==(U const& lhs, value_ptr<T> const& rhs) {
    rhs.this_type_does_not_support_comparisons();	
    return false;	
  }

  template <typename T, typename U>
  inline bool operator!=(U const& lhs, value_ptr<T> const& rhs) {
    rhs.this_type_does_not_support_comparisons();	
    return false;	
  }
}


#endif // FWCoreUtilities_value_ptr_h
