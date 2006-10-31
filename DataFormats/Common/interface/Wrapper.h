#ifndef Common_Wrapper_h
#define Common_Wrapper_h

/*----------------------------------------------------------------------
  
Wrapper: A template wrapper around EDProducts to hold the product ID.

$Id: Wrapper.h,v 1.7 2006/10/30 23:07:52 wmtan Exp $

----------------------------------------------------------------------*/

#include <algorithm>
#include <vector>
#include <memory>
#include <string>

#include "boost/mpl/if.hpp"

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/traits.h"

#include "FWCore/Utilities/interface/GCCPrerequisite.h"

namespace edm {

  template <class T>
  class Wrapper : public EDProduct {
  public:
    typedef T value_type;
    Wrapper() : EDProduct(), present(false), obj() {}
    explicit Wrapper(std::auto_ptr<T> ptr);
    virtual ~Wrapper() {}
    T const * product() const {return (present ? &obj : 0);}
    T const * operator->() const {return product();}
  private:
    virtual bool isPresent_() const {return present;}
    // We wish to disallow copy construction and assignment.
    // We make the copy constructor and assignment operator private.
    Wrapper(Wrapper<T> const& rh); // disallow copy construction
    Wrapper<T> & operator=(Wrapper<T> const&); // disallow assignment
    bool present;
    //   T const obj;
    T obj;
  };

  // This is an attempt to optimize for speed, by avoiding the copying
  // of large objects of type T. In this initial version, we assume
  // that for any class having a 'swap' member function should call
  // 'swap' rather than copying the object.

  template <typename T>
  struct DoSwap
  {
    void operator()(T& a, T& b) { a.swap(b); }
  };

  template <typename T>
  struct DoAssign
  {
    void operator()(T& a, T& b) { a = b; }
  };

  //------------------------------------------------------------
  // Metafunction support for compile-time selection of code used in
  // Wrapper constructor
  //

  namespace detail 
  {

#if GCC_PREREQUISITE(3,4,4)
  //------------------------------------------------------------
  // WHEN WE MOVE to a newer compiler version, the following code
  // should be activated. This code causes compilation failures under
  // GCC 3.2.3, because of a compiler error in dealing with our
  // application of SFINAE. GCC 3.4.2 is known to deal with this code
  // correctly.
  //------------------------------------------------------------
    typedef char (& no_tag )[1]; // type indicating FALSE
    typedef char (& yes_tag)[2]; // type indicating TRUE

    // Definitions for the following struct and function templates are
    // not needed; we only require the declarations.
    template <typename T, void (T::*)(T&)>  struct swap_function;
    template <typename T> no_tag  has_swap_helper(...);
    template <typename T> yes_tag has_swap_helper(swap_function<T, &T::swap> * dummy);

    template<typename T>
    struct has_swap_function
    {
      static bool const value = 
	sizeof(has_swap_helper<T>(0)) == sizeof(yes_tag);
    };
#else
    //------------------------------------------------------------
    // THE FOLLOWING SHOULD BE REMOVED when we move to a newer
    // compiler; see the note above.
    //------------------------------------------------------------
  // has_swap_function is a metafunction of one argument, the type T.
  // As with many metafunctions, it is implemented as a class with a data
  // member 'value', which contains the value 'returned' by the
  // metafunction.
  //
  // has_swap_function<T>::value is 'true' if T has the has_swap
  // member function (with the right signature), and 'false' if T has
  // no such member function.


    template<typename T>
    struct has_swap_function
    {
      static bool const value = has_swap<T>::value;	
    };
#endif
  }

  template <class T>
  Wrapper<T>::Wrapper(std::auto_ptr<T> ptr) :
    EDProduct(), 
    present(ptr.get() != 0),
    obj()
  { 
    if (present) {
       // The following will call swap if T has such a function,
       // and use assignment if T has no such function.
      typename boost::mpl::if_c<detail::has_swap_function<T>::value, 
                                DoSwap<T>, 
                                DoAssign<T> >::type swap_or_assign;
      swap_or_assign(obj, *ptr);	
    }
  }

  std::string
  wrappedClassName(std::string const& className);
}

#endif
