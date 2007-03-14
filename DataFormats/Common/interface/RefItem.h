#ifndef Common_RefItem_h
#define Common_RefItem_h

/*----------------------------------------------------------------------
  
RefItem: Index and pointer to a referenced item.

$Id: RefItem.h,v 1.5 2006/08/22 05:50:16 wmtan Exp $

----------------------------------------------------------------------*/
#include <vector>

#include "DataFormats/Common/interface/RefCore.h"

namespace edm {
  
  template<typename KEY>
  class RefItem {
  public:
    typedef KEY key_type;
    //typedef std::vector<int>::size_type size_type;
    //RefItem() : index_(0), ptr_(refhelper::DefaultHelper<KEY>::defaultValue()) {}
    RefItem() : index_(), ptr_(0) {}
    RefItem(key_type inx, void const* p) : index_(inx), ptr_(p) {}
    ~RefItem() {}
    key_type key() const {return index_;}
    void const *ptr() const {return ptr_;}
    void const *setPtr(void const* p) const {return(ptr_ = p);}
       
private:
    key_type index_;
    mutable void const *ptr_; // transient
  };

  template<typename KEY>
  inline
  bool
  operator==(RefItem<KEY> const& lhs, RefItem<KEY> const& rhs) {
    return lhs.key() == rhs.key();
  }

  template<typename KEY>
  inline
  bool
  operator!=(RefItem<KEY> const& lhs, RefItem<KEY> const& rhs) {
    return !(lhs == rhs);
  }

  template<typename KEY>
  inline
  bool
  operator<(RefItem<KEY> const& lhs, RefItem<KEY> const& rhs) {
    return lhs.key() < rhs.key();
  }

  namespace refitem {
    template< typename C, typename T, typename F, typename KEY>
    struct GetPtrImpl {
      static T const* getPtr_(RefCore const& product, RefItem<KEY> const& item) {
        C const* prod = getProduct<C>(product);
        /*
        typename C::const_iterator it = prod->begin();
         std::advance(it, item.key());
         T const* p = it.operator->();
        */
        F func;
        T const* p = func(*prod, item.key());
        return p;
      }
    };
  }
  template <typename C, typename T, typename F, typename KEY>
  T const* getPtr_(RefCore const& product, RefItem<KEY> const& item) {
     return refitem::GetPtrImpl<C, T, F, KEY>::getPtr_(product, item);
  }

  template <typename C, typename T, typename F, typename KEY>
  inline
  T const* getPtr(RefCore const& product, RefItem<KEY> const& item) {
    T const* p = static_cast<T const *>(item.ptr());
    return (p != 0 ? p : getPtr_<C, T, F>(product, item));
  }
}

#endif
