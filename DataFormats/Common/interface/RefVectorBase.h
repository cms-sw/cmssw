#ifndef Common_RefVectorBase_h
#define Common_RefVectorBase_h

/*----------------------------------------------------------------------
  
RefVectorBase: Base class for a vector of interproduct references.

$Id: RefVectorBase.h,v 1.2 2006/02/13 19:14:22 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/RefBase.h"
#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefItem.h"
#include "DataFormats/Common/interface/ProductID.h"
#include <algorithm>
#include <cstddef>
#include <vector>
#include <sstream>
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {

  class EDProductGetter;
  template <typename T>
  class RefVectorBase {
  public:
    typedef std::vector<RefItem<T> > RefItems;
    typedef typename RefItem<T>::index_type size_type;
    /// Default constructor needed for reading from persistent store. Not for direct use.
    RefVectorBase() : product_(), items_() {}

    explicit RefVectorBase(ProductID const& productID, void const* prodPtr = 0,
                           EDProductGetter const* prodGetter = 0) :
      product_(productID, prodPtr, prodGetter), items_() {}

    /// Destructor
    ~RefVectorBase() {}

    /// Accessor for product ID and product getter
    RefCore const& product() const {return product_;}

    /// Accessor for vector of indexes and pointers
    RefItems const& items() const {return items_;}

    /// Is vector empty?
    bool empty() const {return items_.empty();}

    /// Size of vector
    size_type size() const {return items_.size();}

    void pushBack(RefCore const& prod_, RefItem<T> const& item_) {
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

    /// Capacity of vector
    size_type capacity() const {return items_.capacity();}

    /// Reserve space for vector
    void reserve(size_type n) {items_.reserve(n);}

    /// erase an element from the vector 
    typename RefItems::iterator eraseAtIndex(typename RefItems::size_type index) {
      return items_.erase(items_.begin() + index);
    }
    

  private:
    RefCore product_;
    RefItems items_;
  };

  /// Equality operator
  template< typename T>
  bool
  operator==(RefVectorBase<T> const& lhs, RefVectorBase<T> const& rhs) {
    return lhs.product() == rhs.product() && lhs.items() == rhs.items();
  }

  /// Inequality operator
  template< typename T>
  bool
  operator!=(RefVectorBase<T> const& lhs, RefVectorBase<T> const& rhs) {
    return !(lhs == rhs);
  }
}
#endif
