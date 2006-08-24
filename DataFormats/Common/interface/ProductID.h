#ifndef Common_ProductID_h
#define Common_ProductID_h

/*----------------------------------------------------------------------
  
ProductID: A unique identifier for each EDProduct within a process.

$Id: ProductID.h,v 1.2 2006/08/22 05:50:16 wmtan Exp $

----------------------------------------------------------------------*/

#include <iosfwd>

namespace edm {
  struct ProductID {
    ProductID() : id_(0) {}
    explicit ProductID(unsigned long id) : id_(id) {}
    unsigned long id_;
    bool operator<(ProductID const& rh) const {return id_ < rh.id_;}
    bool operator>(ProductID const& rh) const {return id_ > rh.id_;}
    bool operator==(ProductID const& rh) const {return id_ == rh.id_;}
    bool operator!=(ProductID const& rh) const {return id_ != rh.id_;}
  };

  std::ostream&
  operator<<(std::ostream& os, ProductID const& id);
}
#endif
