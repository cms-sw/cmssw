#ifndef Common_RefItem_h
#define Common_RefItem_h

/*----------------------------------------------------------------------
  
RefItem: Index and pointer to a referenced item.

$Id: RefItem.h,v 1.4 2006/06/14 23:40:33 wmtan Exp $

----------------------------------------------------------------------*/
#include <vector>

#include "DataFormats/Common/interface/RefCore.h"

namespace edm {
  
  template<typename T>
  class RefItem {
  public:
    typedef T key_type;
    //typedef std::vector<int>::size_type size_type;
    //RefItem() : index_(0), ptr_(refhelper::DefaultHelper<T>::defaultValue()) {}
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

  template<typename T>
  inline
  bool
  operator==(RefItem<T> const& lhs, RefItem<T> const& rhs) {
    return lhs.key() == rhs.key();
  }

  template<typename T>
  inline
  bool
  operator!=(RefItem<T> const& lhs, RefItem<T> const& rhs) {
    return !(lhs == rhs);
  }

  template<typename T>
  inline
  bool
  operator<(RefItem<T> const& lhs, RefItem<T> const& rhs) {
    return lhs.key() < rhs.key();
  }

  namespace refitem {
    template< typename C, typename T, typename F, typename TKey>
    struct GetPtrImpl {
      static T const* getPtr_(RefCore const& product, RefItem<TKey> const& item) {
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
  template <typename C, typename T, typename F, typename TKey>
  T const* getPtr_(RefCore const& product, RefItem<TKey> const& item) {
     return refitem::GetPtrImpl<C, T, F, TKey>::getPtr_(product, item);
  }

  template <typename C, typename T, typename F, typename TKey>
  inline
  T const* getPtr(RefCore const& product, RefItem<TKey> const& item) {
    T const* p = static_cast<T const *>(item.ptr());
    return (p != 0 ? p : getPtr_<C, T, F>(product, item));
  }
}

#endif
