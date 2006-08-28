#ifndef Common_Wrapper_h
#define Common_Wrapper_h

/*----------------------------------------------------------------------
  
Wrapper: A template wrapper around EDProducts to hold the product ID.

$Id: Wrapper.h,v 1.3 2006/07/21 21:55:12 wmtan Exp $

----------------------------------------------------------------------*/

#include <vector>
#include <memory>
#include <string>

#include "DataFormats/Common/interface/EDProduct.h"

namespace edm {
  template <class T>
  class Wrapper : public EDProduct {
  public:
    typedef T value_type;
    Wrapper() : EDProduct(), present(false), obj() {}
    explicit Wrapper(std::auto_ptr<T> ptr) :
      EDProduct(), present(ptr.get() != 0), obj(present ? *ptr : T()) {}
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
    T const obj;
  };

  std::string
  wrappedClassName(std::string const& className);
}

#endif
