#ifndef Common_RefVector_h
#define Common_RefVector_h

/*----------------------------------------------------------------------
  
RefVector: A template for a vector of interproduct references.
	Each vector element is a reference to a member of the same product.

$Id: RefVector.h,v 1.18 2006/01/16 09:36:31 llista Exp $

----------------------------------------------------------------------*/

#include <stdexcept>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVectorBase.h"
#include "DataFormats/Common/interface/RefVectorIterator.h"
#include "DataFormats/Common/interface/RefItem.h"
#include "DataFormats/Common/interface/ProductID.h"

namespace edm {

  template <typename C, typename T = typename Ref<C>::value_type>
  class RefVector {
  public:
    typedef RefVectorIterator<C, T> iterator;
    typedef iterator const_iterator;

    typedef T value_type;

    // C is the type of the collection
    // T is the type of a member the collection

    // size_type is the type of the index into the collection
    typedef RefItem::size_type size_type;

    /// Default constructor needed for reading from persistent store. Not for direct use.
    RefVector() : refVector_() {}

    /// Constructor containing product information
    explicit RefVector(ProductID const& productID) :
      refVector_(productID) {}

    /// Destructor
    ~RefVector() {}

    /// Add a Ref<C, T> to the RefVector
    void push_back(Ref<C, T> const& ref) {refVector_.pushBack(ref.ref().product(), ref.ref().item());}

    /// Retrieve an element of the RefVector
    Ref<C, T> const operator[](size_type idx) const {
      RefItem const& item = refVector_.items()[idx];
      RefCore const& product = refVector_.product();
      getPtr<C, T>(product, item);
      return Ref<C, T>(product, item);
    }

    /// Retrieve an element of the RefVector
    Ref<C, T> const at(size_type idx) const {
      RefItem const& item = refVector_.items().at(idx);
      RefCore const& product = refVector_.product();
      getPtr<C, T>(product, item);
      return Ref<C, T>(product, item);
    }

    /// Accessor for all data
    RefVectorBase const& refVector() const {return refVector_;}

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

  private:
    RefVectorBase refVector_;
  };

  template <typename C, typename T>
  inline
  bool
  operator==(RefVector<C, T> const& lhs, RefVector<C, T> const& rhs) {
    return lhs.refVector() == rhs.refVector();
  }

  template <typename C, typename T>
  inline
  bool
  operator!=(RefVector<C, T> const& lhs, RefVector<C, T> const& rhs) {
    return !(lhs == rhs);
  }
}
#endif
