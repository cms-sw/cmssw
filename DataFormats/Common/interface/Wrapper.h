#ifndef Common_Wrapper_h
#define Common_Wrapper_h

/*----------------------------------------------------------------------
  
Wrapper: A template wrapper around EDProducts to hold the product ID.

$Id: Wrapper.h,v 1.8 2005/11/02 06:45:55 wmtan Exp $

----------------------------------------------------------------------*/

#include <vector>
#include <memory>

#include "DataFormats/Common/interface/EDProduct.h"

namespace edm {
  template <class T>
  class Wrapper : public EDProduct {
  public:
    typedef T value_type;
    Wrapper() : EDProduct(), present(false), obj() {}
    explicit Wrapper(std::auto_ptr<T> ptr) :
      EDProduct(), present(ptr.get() != 0), obj(present ? *ptr : T()) {}
    ~Wrapper() {}
    T const * product() const {return (present ? &obj : 0);}
    T const * operator->() const {return product();}
  private:
    // We wish to disallow copy construction and assignment.
    // We make the copy constructor and assignment operator private.
    Wrapper(Wrapper<T> const& rh); // disallow copy construction
    Wrapper<T> & operator=(Wrapper<T> const&); // disallow assignment
    bool present;
    T const obj;
  };
}

#endif
