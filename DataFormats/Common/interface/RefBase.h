#ifndef Common_RefBase_h
#define Common_RefBase_h

/*----------------------------------------------------------------------
  
RefBase: Base class for a single interproduct reference.

$Id: RefBase.h,v 1.16 2005/12/15 23:06:29 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefItem.h"
#include <cstddef>

namespace edm {
  class RefBase {
  public:
    /// Default constructor needed for reading from persistent store. Not for direct use.
    RefBase() : product_(), item_() {}
    /// Destructor
    ~RefBase() {}

    /// Accessor for product ID and product getter.
    RefCore const& product() const;

    /// Accessor for index and pointer
    RefItem const& item() const;

    typedef RefItem::size_type size_type;

    /// General purpose constructor. 
    RefBase(ProductID const& productID, void const* prodPtr, size_type itemIndex,
      void const* itemPtr = 0, EDProductGetter const* prodGetter = 0);

    /// Constructor from RefVector. 
    RefBase(RefCore const& product, RefItem const& item) :
      product_(product), item_(item) {}

  private:
    RefCore product_;
    RefItem item_;
  };

  bool
  operator==(RefBase const& lhs, RefBase const& rhs);

  bool
  operator!=(RefBase const& lhs, RefBase const& rhs);
}
  
#endif
