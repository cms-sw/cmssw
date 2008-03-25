#ifndef DataFormats_Common_RefVectorBase_h
#define DataFormats_Common_RefVectorBase_h

/*----------------------------------------------------------------------
  
RefVectorBase: Base class for a vector of interproduct references.

$Id: RefVectorBase.h,v 1.14 2008/02/15 05:57:03 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Common/interface/RefCore.h"
#include <vector>

namespace edm {

  class EDProductGetter;
  template <typename KEY>
  class RefVectorBase {
  public:
    typedef std::vector<RefItem<KEY> > RefItems;
    typedef KEY key_type;
    typedef typename RefItems::size_type size_type;
    /// Default constructor needed for reading from persistent store. Not for direct use.
    RefVectorBase() : product_(), items_() {}

    explicit RefVectorBase(ProductID const& productID, void const* prodPtr = 0,
                           EDProductGetter const* prodGetter = 0) :
      product_(productID, prodPtr, prodGetter, false), items_() {}

    /// Destructor
    ~RefVectorBase() {}

    /// Accessor for product ID and product getter
    RefCore const& refCore() const {return product_;}

    /// Accessor for vector of keys and pointers
    RefItems const& items() const {return items_;}

    /// Is vector empty?
    bool empty() const {return items_.empty();}

    /// Size of vector
    size_type size() const {return items_.size();}

    void pushBack(RefCore const& product, RefItem<KEY> const& item_) {
      product_.pushBackItem(product, true);
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
    void swap(RefVectorBase<KEY> & other) {
      product_.swap(other.product_);
      items_.swap(other.items_);
    }

    /// Copy assignment
    RefVectorBase& operator=(RefVectorBase const& rhs) {
      RefVectorBase temp(rhs);
      this->swap(temp);
      return *this;
    }

  private:
    RefCore product_;
    RefItems items_;
  };

  /// Equality operator
  template<typename KEY>
  bool
  operator==(RefVectorBase<KEY> const& lhs, RefVectorBase<KEY> const& rhs) {
    return lhs.refCore() == rhs.refCore() && lhs.items() == rhs.items();
  }

  /// Inequality operator
  template<typename KEY>
  bool
  operator!=(RefVectorBase<KEY> const& lhs, RefVectorBase<KEY> const& rhs) {
    return !(lhs == rhs);
  }

  /// swap two vectors
  template<typename KEY>
  inline
  void
  swap(RefVectorBase<KEY> & a, RefVectorBase<KEY> & b) {
    a.swap(b);
  }

}
#endif
