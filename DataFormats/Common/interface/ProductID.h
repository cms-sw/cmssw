#ifndef Common_ProductID_h
#define Common_ProductID_h

/*----------------------------------------------------------------------
  
ProductID: A unique identifier for each EDProduct within a process.

$Id: ProductID.h,v 1.1 2006/02/07 07:01:50 wmtan Exp $

----------------------------------------------------------------------*/

#include <ostream>

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

  inline
  std::ostream&
  operator<<(std::ostream& os, ProductID const& id) {
    os << id.id_;
    return os;
  }
}
#endif
