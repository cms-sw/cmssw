#ifndef FWCoreUtilities_value_ptr_h
#define FWCoreUtilities_value_ptr_h

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
// desireable. Use boost::shared_ptr when sharing of the pointed-to
// object is desireable. Use boost::scoped_ptr when no copying is
// desireable.
//
// The design of value_ptr is taken from Herb Sutter's More
// Exceptional C++, with modifications by Marc Paterno and further
// modifications by Walter Brown. This version is based on the
// ValuePtr found in the Fermilab ZOOM library.
//
// ----------------------------------------------------------------------

#include <algorithm>  // for std::swap()

namespace edm
{
  
  // --------------------------------------------------------------------
  //
  //  Auxiliary traits class template providing default clone()
  //  Users should specialize this template for types that have their
  //  own self-copy operations; failure to do so may lead to slicing!
  //
  // --------------------------------------------------------------------


  template <class T> 
  struct value_ptr_traits  
  {
    static  T *  clone( T const * p )  { return new T( *p ); }
  };

  // --------------------------------------------------------------------
  //
  // Copyable smart pointer class template
  //
  // --------------------------------------------------------------------


  template <class T> 
  class value_ptr  
  {

  public:

    // --------------------------------------------------
    // Default constructor/destructor:
    // --------------------------------------------------

    explicit value_ptr(T* p = 0) : myP(p)  { }
    ~value_ptr()  { delete myP; }

    // --------------------------------------------------
    // Copy constructor/copy assignment:
    // --------------------------------------------------

    value_ptr( value_ptr const & orig ) :
      myP( createFrom(orig.myP))
    { }

    value_ptr &  operator = ( value_ptr const & orig )  
    {
      value_ptr<T>  temp(orig);
      swap(temp);
      return *this;
    }

    // --------------------------------------------------
    // Access mechanisms:
    // --------------------------------------------------

    T&  operator*() const   { return *myP; }
    T*  operator->() const  { return  myP; }

    // --------------------------------------------------
    // Manipulation:
    // --------------------------------------------------

    void  swap( value_ptr & orig )  { std::swap( myP, orig.myP ); }

    // --------------------------------------------------
    // Copy-like construct/assign from compatible value_ptr<>:
    // --------------------------------------------------

    template <class U> 
    value_ptr( value_ptr<U> const & orig ) :
      myP(createFrom(orig.operator->()))
    { }

    template <class U> 
    value_ptr &  operator = ( value_ptr<U> const & orig )  
    {
      value_ptr<T>  temp( orig );
      swap(temp);
      return *this;
    }

  private:

    // --------------------------------------------------
    // Implementation aid:
    // --------------------------------------------------

    template <class U>
    T*
    createFrom( U const * p ) const  
    {
      return p
	? value_ptr_traits<U>::clone( p )
	: 0;
    }

    // --------------------------------------------------
    // Member data:
    // --------------------------------------------------

    T *  myP;

  };  // value_ptr


  // --------------------------------------------------------------------
  //
  // Free-standing swap()
  //
  // --------------------------------------------------------------------

  template <class T> 
  inline
  void  
  swap(  value_ptr<T> & vp1, value_ptr<T> & vp2 ) { vp1.swap( vp2 ); }

}


#endif  // FWCoreUtilities_value_ptr_h
