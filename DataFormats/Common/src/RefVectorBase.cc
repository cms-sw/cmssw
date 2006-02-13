
#include <sstream>
#include "DataFormats/Common/interface/RefVectorBase.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  /// General purpose constructor.  
  RefVectorBase::RefVectorBase(ProductID const& productID, void const* prodPtr,
    EDProductGetter const* prodGetter) :
      product_(productID, prodPtr, prodGetter), items_() {}

  /// Accessor for product and product getter.
  RefCore const&
  RefVectorBase::product() const {
    return product_;
  }

  /// Accessor for index and pointer
  RefVectorBase::RefItems const&
  RefVectorBase::items() const {
    return items_;
  }

  void
  RefVectorBase::pushBack(RefCore const& prod_, RefItem const& item_) {
    if (product_.id() == ProductID()) {
      product_ = prod_; 
    } else if (product_ == prod_) {
	if (product_.productGetter() == 0 && prod_.productGetter() != 0) {
          product_.setProductGetter(prod_.productGetter());
        }
	if (product_.productPtr() == 0 && prod_.productPtr() != 0) {
          product_.setProductPtr(prod_.productPtr());
        }
    } else {
      throw edm::Exception(errors::InvalidReference,"Inconsistency")
	<< "RefVectorBase::push_back: Ref is inconsistent. "
	<< "id = (" << prod_.id() << ") should be (" << product_.id() << ")";
    }
    items_.push_back(item_);
  }

  /// Equality operator
  bool
  operator==(RefVectorBase const& lhs, RefVectorBase const& rhs) {
    return lhs.product() == rhs.product() && lhs.items() == rhs.items();
  }

  /// Inequality operator
  bool
  operator!=(RefVectorBase const& lhs, RefVectorBase const& rhs) {
    return !(lhs == rhs);
  }

  /// erase() an element from the vector
  RefVectorBase::RefItems::iterator  RefVectorBase::eraseAtIndex(RefItems::size_type index) {
    return items_.erase(items_.begin() + index);
  }
}
