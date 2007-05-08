#ifndef Common_Common_h
#define Common_Common_h

/*----------------------------------------------------------------------
  
EDProduct: The base class of all things that will be inserted into the
Event.

$Id: EDProduct.h,v 1.5 2007/01/11 23:39:17 paterno Exp $

----------------------------------------------------------------------*/

#include <vector>
#include "boost/shared_ptr.hpp"

namespace edm {

  class IndirectHolderBaseHelper;

  typedef boost::shared_ptr<IndirectHolderBaseHelper> helper_ptr;

  class EDProduct {
  public:
    EDProduct();
    virtual ~EDProduct();
    bool isPresent() const {return isPresent_();}

    // We have to use vector<void*> to keep the type information out
    // of the EDProduct class.
    void fillView(std::vector<void const*>& view,
		  std::vector<helper_ptr>& helpers) const;

  private:
    // This will never be called.
    // For technical ROOT related reasons, we cannot
    // declare it = 0.
    virtual bool isPresent_() const {return true;}

    virtual void do_fillView(std::vector<void const*>& pointers,
			     std::vector<helper_ptr>& helpers) const = 0;
  };
}
#endif
