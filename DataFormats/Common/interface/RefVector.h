#ifndef Common_RefVector_h
#define Common_RefVector_h

/*----------------------------------------------------------------------
  
RefVector: A template for a vector of interproduct references.
	Each vector element is a reference to a member of the same product.

$Id: RefVector.h,v 1.14 2006/10/28 23:50:34 wmtan Exp $

----------------------------------------------------------------------*/

#include <vector>
#include <stdexcept>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVectorBase.h"
#include "DataFormats/Common/interface/RefVectorIterator.h"
#include "DataFormats/Common/interface/RefItem.h"
#include "DataFormats/Common/interface/ProductID.h"

#include "DataFormats/Common/interface/traits.h"

#include "FWCore/Utilities/interface/GCCPrerequisite.h"

namespace edm {

  template <typename C, typename T = typename Ref<C>::value_type, typename F = typename Ref<C,T>::finder_type>
  class RefVector {
  public:
    typedef RefVectorIterator<C, T, F> iterator;
    typedef iterator const_iterator;

    typedef T value_type;

    // C is the type of the collection
    // T is the type of a member the collection
    
    // key_type is the type of the key into the collextion
    typedef typename Ref<C, T, F>::key_type key_type;
    typedef RefItem<key_type> RefItemType;

    // size_type is the type of the index into the RefVector
    typedef typename std::vector<RefItemType>::size_type size_type;
    

    /// Default constructor needed for reading from persistent store. Not for direct use.
    RefVector() : refVector_() {}

    /// Destructor
    ~RefVector() {}

    /// Add a Ref<C, T> to the RefVector
    void push_back(Ref<C, T, F> const& ref) {refVector_.pushBack(ref.ref().product(), ref.ref().item());}

    /// Retrieve an element of the RefVector
    Ref<C, T, F> const operator[](size_type idx) const {
      RefItemType const& item = refVector_.items()[idx];
      RefCore const& prod = refVector_.product();
      return Ref<C, T, F>(prod, item);
    }

    /// Retrieve an element of the RefVector
    Ref<C, T, F> const at(size_type idx) const {
      RefItemType const& item = refVector_.items().at(idx);
      RefCore const& prod = refVector_.product();
      return Ref<C, T, F>(prod, item);
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

    /// Accessor for product ID.
    ProductID id() const {return refVector_.product().id();}

    /// Checks for null
    bool isNull() const {return id() == ProductID();}

    /// Checks for non-null
    bool isNonnull() const {return !isNull();}

    /// Checks for null
    bool operator!() const {return isNull();}

    /// Accessor for product collection
    // Accessor must get the product if necessary
    C const* product() const {
      return isNull() ? 0 : getProduct<C>(refVector_.product());
    }

    /// Erase an element from the vector.
    iterator erase(iterator const& pos);

    /// Clear the vector.
    void clear() {refVector_.clear();}

    /// Swap two vectors.
    void swap(RefVector<C, T, F> & other);

  private:
    RefVectorBase<key_type> refVector_;
  };

  template <typename C, typename T, typename F>
  inline
  void
  RefVector<C, T, F>::swap(RefVector<C, T, F> & other) {
    refVector_.swap(other.refVector_);
  }

  template <typename C, typename T, typename F>
  inline
  void
  swap(RefVector<C, T, F> & a, RefVector<C, T, F> & b) {
    a.swap(b);
  }

#if ! GCC_PREREQUISITE(3,4,4)
  // has swap function
  template <typename C, typename T, typename F>
  struct has_swap<edm::RefVector<C, T, F> > {
    static bool const value = true;
  };
#endif

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
    RefCore const& prod = refVector_.product();
    return typename RefVector<C, T, F>::iterator(prod, newPos);

  }

}
#endif
