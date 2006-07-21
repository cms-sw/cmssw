#ifndef Common_Common_h
#define Common_Common_h

/*----------------------------------------------------------------------
  
EDProduct: The base class of all things that will be inserted into the
Event.

$Id: EDProduct.h,v 1.1 2006/02/07 07:01:50 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {

  class EDProduct {
  public:
    EDProduct();
    virtual ~EDProduct();
    bool isPresent() const {return isPresent_();}
  private:
    virtual bool isPresent_() const = 0;
  };
}
#endif
