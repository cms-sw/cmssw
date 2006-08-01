#ifndef Common_Common_h
#define Common_Common_h

/*----------------------------------------------------------------------
  
EDProduct: The base class of all things that will be inserted into the
Event.

$Id: EDProduct.h,v 1.2 2006/07/21 21:55:12 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {

  class EDProduct {
  public:
    EDProduct();
    virtual ~EDProduct();
    bool isPresent() const {return isPresent_();}
  private:
    // This will never be called.
    // For technical ROOT related reasons, we cannot
    // declare it = 0.
    virtual bool isPresent_() const {return true;}
  };
}
#endif
