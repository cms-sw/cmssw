#ifndef Common_RefVector_h
#define Common_RefVector_h

/*----------------------------------------------------------------------
  
RefVector: A template for a vector of interproduct references.
	Each vector element is a reference to a member of the same product.

$Id: RefVector.h,v 1.4 2006/04/28 23:02:39 wmtan Exp $

----------------------------------------------------------------------*/

#include <vector>
#include <stdexcept>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVectorBase.h"
#include "DataFormats/Common/interface/RefVectorIterator.h"
#include "DataFormats/Common/interface/RefItem.h"
#include "DataFormats/Common/interface/ProductID.h"

namespace edm {

  template <typename C, typename T = typename Ref<C>::value_type, typename F = typename Ref<C>::finder_type>
  class RefVector {
  public:
    typedef RefVectorIterator<C, T, F> iterator;
    typedef iterator const_iterator;

    typedef T value_type;

    // C is the type of the collection
    // T is the type of a member the collection
    
    // key_type is the type of the key into the collextion
    typedef typename Ref<C, T, F>::index_type key_type;
    typedef RefItem<key_type> RefItemType;

    // size_type is the type of the index into the RefVector
    typedef typename std::vector<RefItemType>::size_type size_type;
    

    /// Default constructor needed for reading from persistent store. Not for direct use.
    RefVector() : refVector_() {}

    /// Constructor containing product information
    explicit RefVector(ProductID const& productID) :
      refVector_(productID) {}

    /// Destructor
    ~RefVector() {}

    /// Add a Ref<C, T> to the RefVector
    void push_back(Ref<C, T, F> const& ref) {refVector_.pushBack(ref.ref().product(), ref.ref().item());}

    /// Retrieve an element of the RefVector
    Ref<C, T, F> const operator[](size_type idx) const {
      RefItemType const& item = refVector_.items()[idx];
      RefCore const& product = refVector_.product();
      return Ref<C, T, F>(product, item);
    }

    /// Retrieve an element of the RefVector
    Ref<C, T, F> const at(size_type idx) const {
      RefItemType const& item = refVector_.items().at(idx);
      RefCore const& product = refVector_.product();
      return Ref<C, T, F>(product, item);
    }

    /// Accessor for all data
    RefVectorBase<key_type> const& refVector() const {return refVector_;}

    /// Is the RefVector empty
    bool empty() const {return refVector_.empty();}

    /// Size of the RefVector
    size_type size() const {return refVector_.size();}

    /// Capacity of the RefVector
    size_type capacity() const {return refVector_.capacity();}

    /// Reserve space for RefVector
    void reserve(size_type n) {refVector_.reserve(n);}

    /// Initialize an iterator over the RefVector
    iterator begin() const {return iterator(refVector_.product(), refVector_.items().begin());}

    /// Termination of iteration
    iterator end() const {return iterator(refVector_.product(), refVector_.items().end());}

    /// Accessor for product collection
    C const* product() const {return static_cast<C const *>(refVector_.product().productPtr());}

    /// Erase an element from the vector.
    iterator erase(iterator const& pos);

  private:
    RefVectorBase<key_type> refVector_;
  };

  template <typename C, typename T, typename F>
  inline
  bool
  operator==(RefVector<C, T, F> const& lhs, RefVector<C, T, F> const& rhs) {
    return lhs.refVector() == rhs.refVector();
  }

  template <typename C, typename T, typename F>
  inline
  bool
  operator!=(RefVector<C, T, F> const& lhs, RefVector<C, T, F> const& rhs) {
    return !(lhs == rhs);
  }

  template <typename C, typename T, typename F>
  inline
  typename RefVector<C, T, F>::iterator RefVector<C, T, F>::erase(iterator const& pos) {
    typename RefVectorBase<key_type>::RefItems::size_type index = pos - begin();
    typename RefVectorBase<key_type>::RefItems::iterator newPos = refVector_.eraseAtIndex(index);
    RefCore const& product = refVector_.product();
    return RefVector<C, T, F>::iterator(product, newPos);

  }

}
#endif
