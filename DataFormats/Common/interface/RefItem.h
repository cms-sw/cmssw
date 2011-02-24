#ifndef DataFormats_Common_RefItem_h
#define DataFormats_Common_RefItem_h

#error "this file should not be included anymore"

/*----------------------------------------------------------------------
  
RefItem: Index and pointer to a referenced item.

$Id: RefItem.h,v 1.11.6.1 2011/02/17 03:08:31 chrjones Exp $

----------------------------------------------------------------------*/
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/ConstPtrCache.h"

namespace edm {
  
  template<typename KEY>
  class RefItem {
  public:
    typedef KEY key_type;

    RefItem() : index_(key_traits<key_type>::value), cache_(0) {}

    RefItem(key_type inx, void const* p) : index_(inx), cache_(p) {}

    ~RefItem() {}

    key_type key() const {return index_;}
    void const *ptr() const {return cache_.ptr_;}
    void const *setPtr(void const* p) const {return(cache_.ptr_ = p);}

    bool isValid() const { return index_!=edm::key_traits<key_type>::value; }
    bool isNonnull() const { return isValid(); }
    bool isNull() const { return !isValid(); }
       
private:
    key_type index_;
    mutable ConstPtrCache cache_; //Type handles the transient
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
