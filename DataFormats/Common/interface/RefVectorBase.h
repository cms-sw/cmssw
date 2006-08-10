#ifndef Common_RefVectorBase_h
#define Common_RefVectorBase_h

/*----------------------------------------------------------------------
  
RefVectorBase: Base class for a vector of interproduct references.

$Id: RefVectorBase.h,v 1.6 2006/06/14 23:26:30 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefItem.h"
#include "DataFormats/Common/interface/ProductID.h"
#include <vector>

namespace edm {

  class EDProductGetter;
  template <typename T>
  class RefVectorBase {
  public:
    typedef std::vector<RefItem<T> > RefItems;
    typedef T key_type;
    typedef typename RefItems::size_type size_type;
    /// Default constructor needed for reading from persistent store. Not for direct use.
    RefVectorBase() : product_(), items_() {}

    explicit RefVectorBase(ProductID const& productID, void const* prodPtr = 0,
                           EDProductGetter const* prodGetter = 0) :
      product_(productID, prodPtr, prodGetter), items_() {}

    /// Destructor
    ~RefVectorBase() {}

    /// Accessor for product ID and product getter
    RefCore const& product() const {return product_;}

    /// Accessor for vector of keys and pointers
    RefItems const& items() const {return items_;}

    /// Is vector empty?
    bool empty() const {return items_.empty();}

    /// Size of vector
    size_type size() const {return items_.size();}

    void pushBack(RefCore const& prod_, RefItem<T> const& item_) {
      checkProduct(prod_, product_);
      items_.push_back(item_);
    }

    /// Capacity of vector
    size_type capacity() const {return items_.capacity();}

    /// Reserve space for vector
    void reserve(size_type n) {items_.reserve(n);}

    /// erase an element from the vector 
    typename RefItems::iterator eraseAtIndex(size_type index) {
      return items_.erase(items_.begin() + index);
    }
    
    /// clear the vector
    void clear() {
      items_.clear();
      product_ = RefCore();
    }

    /// swap two vectors
    void swap(RefVectorBase<T> & other) {
      std::swap(product_, other.product_);
      items_.swap(other.items_);
    }

  private:
    RefCore product_;
    RefItems items_;
  };

  /// Equality operator
  template<typename T>
  bool
  operator==(RefVectorBase<T> const& lhs, RefVectorBase<T> const& rhs) {
    return lhs.product() == rhs.product() && lhs.items() == rhs.items();
  }

  /// Inequality operator
  template<typename T>
  bool
  operator!=(RefVectorBase<T> const& lhs, RefVectorBase<T> const& rhs) {
    return !(lhs == rhs);
  }

  /// swap two vectors
  template<typename T>
  inline
  void
  swap(RefVectorBase<T> & a, RefVectorBase<T> & b) {
    a.swap(b);
  }

}
#endif
