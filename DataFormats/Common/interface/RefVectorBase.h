#ifndef DataFormats_Common_RefVectorBase_h
#define DataFormats_Common_RefVectorBase_h

/*----------------------------------------------------------------------
  
RefVectorBase: Base class for a vector of interproduct references.


----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Common/interface/RefCore.h"
#include <vector>
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

namespace edm {

  class EDProductGetter;
  template <typename KEY>
  class RefVectorBase {
  public:
    typedef std::vector<KEY > keys_type;
    typedef KEY key_type;
    typedef typename keys_type::size_type size_type;
    /// Default constructor needed for reading from persistent store. Not for direct use.
    RefVectorBase() : product_(), keys_() {}
    RefVectorBase( RefVectorBase const & rhs) : product_(rhs.product_), keys_(rhs.keys_) {}
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
    RefVectorBase( RefVectorBase && rhs)  noexcept : product_(std::move(rhs.product_)), keys_(std::move(rhs.keys_)) {}
#endif

    explicit RefVectorBase(ProductID const& productID, void const* prodPtr = 0,
                           EDProductGetter const* prodGetter = 0) :
      product_(productID, prodPtr, prodGetter, false), keys_() {}

    /// Destructor
    ~RefVectorBase() noexcept {}

    /// Accessor for product ID and product getter
    RefCore const& refCore() const {return product_;}

    /// Accessor for vector of keys and pointers
    keys_type const& keys() const {return keys_;}

    /// Is vector empty?
    bool empty() const {return keys_.empty();}

    /// Size of vector
    size_type size() const {return keys_.size();}

    void pushBack(RefCore const& product, KEY const& key) {
      product_.pushBackItem(product, true);
      keys_.push_back(key);
    }

    /// Capacity of vector
    size_type capacity() const {return keys_.capacity();}

    /// Reserve space for vector
    void reserve(size_type n) {keys_.reserve(n);}

    /// erase an element from the vector 
    typename keys_type::iterator eraseAtIndex(size_type index) {
      return keys_.erase(keys_.begin() + index);
    }
    
    /// clear the vector
    void clear() {
      keys_.clear();
      product_ = RefCore();
    }

    /// swap two vectors
    void swap(RefVectorBase<KEY> & other)  noexcept {
      product_.swap(other.product_);
      keys_.swap(other.keys_);
    }

    /// Copy assignment
    RefVectorBase& operator=(RefVectorBase const& rhs) {
      RefVectorBase temp(rhs);
      this->swap(temp);
      return *this;
    }
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
    RefVectorBase& operator=(RefVectorBase && rhs)  noexcept {
      product_ = std::move(rhs.product_); 
      keys_ =std::move(rhs.keys_);
      return *this;
    }
#endif

    //Needed for ROOT storage
    CMS_CLASS_VERSION(11)

  private:
    RefCore product_;
    keys_type keys_;
  };

  /// Equality operator
  template<typename KEY>
  bool
  operator==(RefVectorBase<KEY> const& lhs, RefVectorBase<KEY> const& rhs) {
    return lhs.refCore() == rhs.refCore() && lhs.keys() == rhs.keys();
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
