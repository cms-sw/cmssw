#ifndef Common_ProductID_h
#define Common_ProductID_h

/*----------------------------------------------------------------------
  
ProductID: A unique identifier for each EDProduct within a process.

$Id: ProductID.h,v 1.3 2006/08/24 22:15:44 wmtan Exp $

----------------------------------------------------------------------*/

#include <iosfwd>

namespace edm {
  struct ProductID {
    ProductID() : id_(0) {}
    explicit ProductID(unsigned int id) : id_(id) {}
    unsigned int id_;
    bool operator<(ProductID const& rh) const {return id_ < rh.id_;}
    bool operator>(ProductID const& rh) const {return id_ > rh.id_;}
    bool operator==(ProductID const& rh) const {return id_ == rh.id_;}
    bool operator!=(ProductID const& rh) const {return id_ != rh.id_;}
  };

  std::ostream&
  operator<<(std::ostream& os, ProductID const& id);
}
#endif
