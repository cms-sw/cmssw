#ifndef Common_Common_h
#define Common_Common_h

/*----------------------------------------------------------------------
  
EDProduct: The base class of all things that will be inserted into the
Event.

$Id: EDProduct.h,v 1.4 2006/12/28 18:51:02 paterno Exp $

----------------------------------------------------------------------*/

#include <vector>

namespace edm {

  class EDProduct {
  public:
    EDProduct();
    virtual ~EDProduct();
    bool isPresent() const {return isPresent_();}

    // We have to use vector<void*> to keep the type information out
    // of the EDProduct class.
    void fillView(std::vector<void const*>& view) const;

  private:
    // This will never be called.
    // For technical ROOT related reasons, we cannot
    // declare it = 0.
    virtual bool isPresent_() const {return true;}

    virtual void do_fillView(std::vector<void const*>& pointers) const = 0;
  };
}
#endif
