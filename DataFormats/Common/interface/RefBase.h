#ifndef Common_RefBase_h
#define Common_RefBase_h

/*----------------------------------------------------------------------
  
RefBase: Base class for a single interproduct reference.

$Id: RefBase.h,v 1.5 2006/12/16 03:41:25 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefItem.h"
#include <cstddef>

namespace edm {

  template<typename KEY>
  class RefBase {
  public:
    /// Default constructor needed for reading from persistent store. Not for direct use.
    RefBase() : product_(), item_() {}
    /// Destructor
    ~RefBase() {}

    /// Accessor for product ID and product getter.
    RefCore const& refCore() const { return product_;}

    /// Accessor for index and pointer
    RefItem<KEY> const& item() const {return item_;}

    typedef typename RefItem<KEY>::key_type key_type;

    /// General purpose constructor. 
    RefBase(ProductID const& productID, void const* prodPtr, key_type itemKey,
            void const* itemPtr = 0, EDProductGetter const* prodGetter = 0):
      product_(productID, prodPtr, prodGetter), item_(itemKey, itemPtr) {}

    /// Constructor from RefVector. 
    RefBase(RefCore const& prod, RefItem<KEY> const& itm) :
      product_(prod), item_(itm) {}

  private:
    RefCore product_;
    RefItem<KEY> item_;
  };

  template <typename KEY>
  bool
  operator==(RefBase<KEY> const& lhs, RefBase<KEY> const& rhs) {
    return lhs.refCore() == rhs.refCore() && lhs.item() == rhs.item();
  }
  
  template <typename KEY>
  bool
  operator!=(RefBase<KEY> const& lhs, RefBase<KEY> const& rhs) {
    return !(lhs == rhs);
  }

  template <typename KEY>
  bool
  operator<(RefBase<KEY> const& lhs, RefBase<KEY> const& rhs) {
    return (lhs.refCore() == rhs.refCore() ?  lhs.item() < rhs.item() : lhs.refCore() < rhs.refCore());
  }
}
  
#endif
