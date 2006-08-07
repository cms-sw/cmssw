#ifndef Common_Wrapper_h
#define Common_Wrapper_h

/*----------------------------------------------------------------------
  
Wrapper: A template wrapper around EDProducts to hold the product ID.

$Id: Wrapper.h,v 1.4 2006/07/21 22:29:40 wmtan Exp $

----------------------------------------------------------------------*/

#include <algorithm>
#include <vector>
#include <memory>
#include <string>

#include "boost/mpl/if.hpp"

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/traits.h"

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

  template <class T>
  struct DoSwap
  {
    void operator()(T& a, T& b) { a.swap(b); }
  };

  template <class T>
  struct DoAssign
  {
    void operator()(T& a, T& b) { a = b; }
  };

  template <class T>
  Wrapper<T>::Wrapper(std::auto_ptr<T> ptr) :
    EDProduct(), 
    present(ptr.get() != 0),
    obj()
  { 
    if (present) {
      // When we move to GCC 3.4, get rid of has_swap<T> trait
      // and do this with metaprogramming; see Event.h for an example.
      typename boost::mpl::if_c<has_swap<T>::value, 
                                DoSwap<T>, 
                                DoAssign<T> 
                               >::type       swap_or_assign;

      swap_or_assign(obj, *ptr);	
    }
  }

  std::string
  wrappedClassName(std::string const& className);
}

#endif
