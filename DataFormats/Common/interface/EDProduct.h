#ifndef Common_Common_h
#define Common_Common_h

/*----------------------------------------------------------------------
  
EDProduct: The base class of all things that will be inserted into the
Event.

$Id: EDProduct.h,v 1.6 2005/10/11 21:32:24 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {

  class EDProduct {
  public:
    EDProduct();
    virtual ~EDProduct();
  };
}
#endif
