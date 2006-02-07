#ifndef Common_RefVectorBase_h
#define Common_RefVectorBase_h

/*----------------------------------------------------------------------
  
RefVectorBase: Base class for a vector of interproduct references.

$Id: RefVectorBase.h,v 1.14 2005/12/16 00:37:43 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/RefBase.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefItem.h"
#include "DataFormats/Common/interface/ProductID.h"
#include <algorithm>
#include <cstddef>
#include <vector>

namespace edm {

  class EDProductGetter;
  class RefVectorBase {
  public:
    typedef std::vector<RefItem> RefItems;
    typedef RefItem::size_type size_type;
    /// Default constructor needed for reading from persistent store. Not for direct use.
    RefVectorBase() : product_(), items_() {}

    explicit RefVectorBase(ProductID const& productID, void const* prodPtr = 0,
      EDProductGetter const* prodGetter = 0);

    /// Destructor
    ~RefVectorBase() {}

    /// Accessor for product ID and product getter
    RefCore const& product() const;

    /// Accessor for vector of indexes and pointers
    RefItems const& items() const;

    /// Is vector empty?
    bool empty() const {return items_.empty();}

    /// Size of vector
    size_type size() const {return items_.size();}

    void pushBack(RefCore const& prod_, RefItem const& item_);

    /// Capacity of vector
    size_type capacity() const {return items_.capacity();}

    /// Reserve space for vector
    void reserve(size_type n) {items_.reserve(n);}

  private:
    RefCore product_;
    RefItems items_;
  };

  /// Equality operator
  bool
  operator==(RefVectorBase const& lhs, RefVectorBase const& rhs);

  /// Inequality operator
  bool
  operator!=(RefVectorBase const& lhs, RefVectorBase const& rhs);
}
#endif
