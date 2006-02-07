#ifndef Common_RefItem_h
#define Common_RefItem_h

/*----------------------------------------------------------------------
  
RefItem: Index and pointer to a referenced item.

$Id: RefItem.h,v 1.8 2005/12/15 23:06:29 wmtan Exp $

----------------------------------------------------------------------*/
#include <vector>
#include <iostream>

#include "DataFormats/Common/interface/RefCore.h"

namespace edm {

  class RefItem  {
  public:
    typedef std::vector<int>::size_type size_type;
    RefItem() : index_(0), ptr_(0) {}
    RefItem(size_type inx, void const* p) : index_(inx), ptr_(p) {}
    ~RefItem() {}
    unsigned long index() const {return index_;}
    void const *ptr() const {return ptr_;}
    void const *setPtr(void const* p) const;
  private:
    size_type index_;
    mutable void const *ptr_; // transient
  };

  inline
  bool
  operator==(RefItem const& lhs, RefItem const& rhs) {
    return lhs.index() == rhs.index();
  }

  inline
  bool
  operator!=(RefItem const& lhs, RefItem const& rhs) {
    return !(lhs == rhs);
  }

  template <typename C, typename T>
  T const* getPtr_(RefCore const& product, RefItem const& item) {
    C const* prod = getProduct<C>(product);
    typename C::const_iterator it = prod->begin();
    std::advance(it, item.index());
    T const* p = it.operator->();
    return p;
  }

  template <typename C, typename T>
  inline
  T const* getPtr(RefCore const& product, RefItem const& item) {
    T const* p = static_cast<T const *>(item.ptr());
    return (p != 0 ? p : getPtr_<C, T>(product, item));
  }
}

#endif
