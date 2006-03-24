#ifndef Common_RefItem_h
#define Common_RefItem_h

/*----------------------------------------------------------------------
  
RefItem: Index and pointer to a referenced item.

$Id: RefItem.h,v 1.2 2006/03/03 17:00:22 chrjones Exp $

----------------------------------------------------------------------*/
#include <vector>
#include <iostream>

#include "DataFormats/Common/interface/RefCore.h"

namespace edm {
  
  template<typename T>
  class RefItem {
  public:
    typedef T index_type;
    //typedef std::vector<int>::size_type size_type;
    //RefItem() : index_(0), ptr_(refhelper::DefaultHelper<T>::defaultValue()) {}
    RefItem() : index_(), ptr_(0) {}
    RefItem(index_type inx, void const* p) : index_(inx), ptr_(p) {}
    ~RefItem() {}
    index_type index() const {return index_;}
    void const *ptr() const {return ptr_;}
    void const *setPtr(void const* p) const {return(ptr_ = p);}
       
private:
    index_type index_;
    mutable void const *ptr_; // transient
  };

  template<typename T>
  inline
  bool
  operator==(RefItem<T> const& lhs, RefItem<T> const& rhs) {
    return lhs.index() == rhs.index();
  }

  template<typename T>
  inline
  bool
  operator!=(RefItem<T> const& lhs, RefItem<T> const& rhs) {
    return !(lhs == rhs);
  }

  namespace refitem {
    template< typename C, typename T, typename F, typename TIndex>
    struct GetPtrImpl {
      static T const* getPtr_(RefCore const& product, RefItem<TIndex> const& item) {
        C const* prod = getProduct<C>(product);
        /*
        typename C::const_iterator it = prod->begin();
         std::advance(it, item.index());
         T const* p = it.operator->();
        */
        F func;
        T const* p = func(*prod, item.index());
        return p;
      }
    };
  }
  template <typename C, typename T, typename F, typename TIndex>
  T const* getPtr_(RefCore const& product, RefItem<TIndex> const& item) {
     return refitem::GetPtrImpl<C, T, F, TIndex>::getPtr_(product, item);
  }

  template <typename C, typename T, typename F, typename TIndex>
  inline
  T const* getPtr(RefCore const& product, RefItem<TIndex> const& item) {
    T const* p = static_cast<T const *>(item.ptr());
    return (p != 0 ? p : getPtr_<C, T, F>(product, item));
  }
}

#endif
