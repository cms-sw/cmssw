#ifndef Common_ProductID_h
#define Common_ProductID_h

/*----------------------------------------------------------------------
  
ProductID: A unique identifier for each EDProduct within a process.

$Id: ProductID.h,v 1.4 2006/12/18 19:15:18 wmtan Exp $

----------------------------------------------------------------------*/

#include <iosfwd>

namespace edm {
  struct ProductID {
    ProductID() : id_(0) {}
    explicit ProductID(unsigned int id) : id_(id) {}
    bool isValid() const { return id_ != 0U; }
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
