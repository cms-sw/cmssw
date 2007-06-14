#ifndef DataFormats_Common_RefItem_h
#define DataFormats_Common_RefItem_h

/*----------------------------------------------------------------------
  
RefItem: Index and pointer to a referenced item.

$Id: RefItem.h,v 1.9 2007/05/15 17:10:24 wmtan Exp $

----------------------------------------------------------------------*/
#include "DataFormats/Common/interface/traits.h"

namespace edm {
  
  template<typename KEY>
  class RefItem {
  public:
    typedef KEY key_type;

    RefItem() : index_(key_traits<key_type>::value), ptr_(0) {}

    RefItem(key_type inx, void const* p) : index_(inx), ptr_(p) {}

    ~RefItem() {}

    key_type key() const {return index_;}
    void const *ptr() const {return ptr_;}
    void const *setPtr(void const* p) const {return(ptr_ = p);}

    bool isValid() const { return index_!=edm::key_traits<key_type>::value; }
    bool isNonnull() const { return isValid(); }
    bool isNull() const { return !isValid(); }
       
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
}

#endif
