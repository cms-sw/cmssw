#ifndef DataFormats_Common_RefVector_h
#define DataFormats_Common_RefVector_h

/*----------------------------------------------------------------------

RefVector: A template for a vector of interproduct references.
        Each vector element is a reference to a member of the same product.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Common/interface/FillView.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefHolderBase.h"
#include "DataFormats/Common/interface/RefTraits.h"
#include "DataFormats/Common/interface/RefVectorBase.h"
#include "DataFormats/Common/interface/RefVectorIterator.h"
#include "DataFormats/Common/interface/RefVectorTraits.h"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace edm {

  template<typename C,
            typename T = typename refhelper::ValueTrait<C>::value,
            typename F = typename refhelper::FindTrait<C, T>::value>
  class RefVector {
  public:
    typedef C                               collection_type;
    typedef T                               member_type;
    typedef F                               finder_type;
    typedef typename refhelper::RefVectorTrait<C, T, F>::iterator_type iterator;
    typedef iterator                        const_iterator;
    typedef typename refhelper::RefVectorTrait<C, T, F>::ref_type value_type;
    typedef value_type const                const_reference; // better this than the default 'const value_type &'
    typedef const_reference                 reference;       // as operator[] returns 'const R' and not 'R &'

    // key_type is the type of the key into the collection
    typedef typename value_type::key_type   key_type;
    typedef std::vector<key_type>           KeyVec;

    // size_type is the type of the index into the RefVector
    typedef typename KeyVec::size_type      size_type;
    typedef RefVectorBase<key_type>         contents_type;

    /// Default constructor needed for reading from persistent
    /// store. Not for direct use.
    RefVector() : refVector_() {}
    RefVector(RefVector const& rh) : refVector_(rh.refVector_){}
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
    RefVector(RefVector && rh)  noexcept : refVector_(std::move(rh.refVector_)){}
#endif

    RefVector(ProductID const& iId) : refVector_(iId) {}
    /// Add a Ref<C, T> to the RefVector
    void push_back(value_type const& ref) {
      refVector_.pushBack(ref.refCore(), ref.key());
    }

    /// Retrieve an element of the RefVector
    value_type const operator[](size_type idx) const {
      RefCore const& core = refVector_.refCore();
      key_type const& key = refVector_.keys()[idx];
      void const* memberPointer = refVector_.cachedMemberPointer(idx);
      if(memberPointer) {
        RefCore newCore(core);
        newCore.setProductPtr(memberPointer);
        return value_type(newCore, key);
      }
      return value_type(core, key);
    }

    /// Retrieve an element of the RefVector
    value_type const at(size_type idx) const {
      RefCore const& core = refVector_.refCore();
      key_type const& key = refVector_.keys().at(idx);
      void const* memberPointer = refVector_.cachedMemberPointer(idx);
      if(memberPointer) {
        RefCore newCore(core);
        newCore.setProductPtr(memberPointer);
        return value_type(newCore, key);
      }
      return value_type(core, key);
    }

    /// Accessor for all data
    contents_type const& refVector() const {return refVector_;}

    /// Is the RefVector empty
    bool empty() const {return refVector_.empty();}

    /// Size of the RefVector
    size_type size() const {return refVector_.size();}

    /// Capacity of the RefVector
    size_type capacity() const {return refVector_.capacity();}

    /// Reserve space for RefVector
    void reserve(size_type n) {refVector_.reserve(n);}

    /// Initialize an iterator over the RefVector
    const_iterator begin() const;

    /// Termination of iteration
    const_iterator end() const;

    /// Accessor for product ID.
    ProductID id() const {return refVector_.refCore().id();}

    /// Accessor for product getter
    EDProductGetter const* productGetter() const {return refVector_.refCore().productGetter();}

    /// Checks for null
    bool isNull() const {return !id().isValid();}

    /// Checks for non-null
    bool isNonnull() const {return !isNull();}

    /// Checks for null
    bool operator!() const {return isNull();}

    /// Checks if product collection is in memory or available
    /// in the Event. No type checking is done.
    bool isAvailable() const;

    /// Checks if product collection is tansient (i.e. non persistable)
    bool isTransient() const {return refVector_.refCore().isTransient();}

    /// Erase an element from the vector.
    iterator erase(iterator const& pos);

    /// Clear the vector.
    void clear() {refVector_.clear();}

    /// Swap two vectors.
    void swap(RefVector<C, T, F> & other)  noexcept;

    /// Copy assignment.
    RefVector& operator=(RefVector const& rhs);
#if defined(__GXX_EXPERIMENTAL_CXX0X__)
    RefVector& operator=(RefVector  && rhs)  noexcept { 
      refVector_ = std::move(rhs.refVector_);
      return *this;
    }
#endif
    /// Checks if product is in memory.
    bool hasProductCache() const {return refVector_.refCore().productPtr() != 0;}

    void fillView(ProductID const& id,
                  std::vector<void const*>& pointers,
                  FillViewHelperVector& helpers) const;

    //Needed for ROOT storage
    CMS_CLASS_VERSION(13)

  private:

    contents_type refVector_;
  };

  template<typename C, typename T, typename F>
  inline
  void
  RefVector<C, T, F>::swap(RefVector<C, T, F> & other) noexcept {
    refVector_.swap(other.refVector_);
  }

  template<typename C, typename T, typename F>
  inline
  RefVector<C, T, F>&
  RefVector<C, T, F>::operator=(RefVector<C, T, F> const& rhs) {
    RefVector<C, T, F> temp(rhs);
    this->swap(temp);
    return *this;
  }

  template<typename C, typename T, typename F>
  inline
  void
  swap(RefVector<C, T, F> & a, RefVector<C, T, F> & b)  noexcept {
    a.swap(b);
  }

#if defined(__GXX_EXPERIMENTAL_CXX0X__)
  template<typename C, typename T, typename F>
  void
  RefVector<C,T,F>::fillView(ProductID const&,
                             std::vector<void const*>& pointers,
                             FillViewHelperVector& helpers) const {

    pointers.reserve(this->size());
    helpers.reserve(this->size());

    size_type key = 0;
    for(const_iterator i=begin(), e=end(); i!=e; ++i, ++key) {
      member_type const* address = i->isNull() ? 0 : &**i;
      pointers.push_back(address);
      helpers.emplace_back(i->id(),i->key());
    }
  }
#endif

  template<typename C, typename T, typename F>
  inline
  void
  fillView(RefVector<C,T,F> const& obj,
           ProductID const& id,
           std::vector<void const*>& pointers,
           FillViewHelperVector& helpers) {
    obj.fillView(id, pointers, helpers);
  }

  template<typename C, typename T, typename F>
  struct has_fillView<RefVector<C,T,F> > {
    static bool const value = true;
  };

  template<typename C, typename T, typename F>
  inline
  bool
  operator==(RefVector<C, T, F> const& lhs, RefVector<C, T, F> const& rhs) {
    return lhs.refVector() == rhs.refVector();
  }

  template<typename C, typename T, typename F>
  inline
  bool
  operator!=(RefVector<C, T, F> const& lhs, RefVector<C, T, F> const& rhs) {
    return !(lhs == rhs);
  }

  template<typename C, typename T, typename F>
  inline
  typename RefVector<C, T, F>::iterator
  RefVector<C, T, F>::erase(iterator const& pos) {
    typename contents_type::keys_type::size_type index = pos - begin();
    typename contents_type::keys_type::iterator newPos =
      refVector_.eraseAtIndex(index);
    typename contents_type::keys_type::size_type newIndex = newPos - refVector_.keys().begin();
    return iterator(this, newIndex);
  }

  template<typename C, typename T, typename F>
  typename RefVector<C, T, F>::const_iterator RefVector<C, T, F>::begin() const {
    return iterator(this, 0);
  }

  template<typename C, typename T, typename F>
  typename RefVector<C, T, F>::const_iterator RefVector<C, T, F>::end() const {
    return iterator(this, size());
  }

  template<typename C, typename T, typename F>
  bool
  RefVector<C, T, F>::isAvailable() const {
    if(refVector_.refCore().isAvailable()) {
      return true;
    } else if(empty()) {
      return false;
    }

    // The following is the simplest implementation, but
    // this is woefully inefficient and could be optimized
    // to run much faster with some nontrivial effort. There
    // is only 1 place in the code base where this function
    // is used at all and I'm not sure whether it will ever
    // be used with thinned collections, so for the moment I
    // am not spending the time to optimize this.
    for(size_type i = 0; i < size(); ++i) {
      if(!(*this)[i].isAvailable()) {
        return false;
      }
    }
    return true;
  }

  template<typename C, typename T, typename F>
  std::ostream&
  operator<<(std::ostream& os, RefVector<C,T,F> const& r) {
    for(typename RefVector<C,T,F>::const_iterator
           i = r.begin(),
           e = r.end();
           i != e;
           ++i) {
        os << *i << '\n';
      }
    return os;
  }
}

#include "DataFormats/Common/interface/GetProduct.h"
namespace edm {
  namespace detail {

    template<typename C, typename T, typename F>
    struct GetProduct<RefVector<C, T, F> > {
      typedef T element_type;
      typedef typename RefVector<C, T, F>::const_iterator iter;
      static element_type const* address(iter const& i) {
        return &**i;
      }
    };
  }
}
#endif
