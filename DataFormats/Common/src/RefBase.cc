#include "DataFormats/Common/interface/RefBase.h"

namespace edm {
  /// General purpose constructor. 
  RefBase::RefBase(ProductID const& productID, void const* prodPtr, size_type itemIndex,
    void const* itemPtr, EDProductGetter const* prodGetter) :
    product_(productID, prodPtr, prodGetter), item_(itemIndex, itemPtr) {}

  /// Accessor for product and product getter.
  RefCore const&
  RefBase:: product() const {
    return product_;
  }

  /// Accessor for index and pointer
  RefItem const&
  RefBase::item() const {
    return item_;
  }

  bool
  operator==(RefBase const& lhs, RefBase const& rhs) {
    return lhs.product() == rhs.product() && lhs.item() == rhs.item();
  }

  bool
  operator!=(RefBase const& lhs, RefBase const& rhs) {
    return !(lhs == rhs);
  }
}
