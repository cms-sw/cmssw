#ifndef DataFormats_Common_RefBase_h
#define DataFormats_Common_RefBase_h

/*----------------------------------------------------------------------
  
RefBase: Base class for a single interproduct reference.

$Id: RefBase.h,v 1.10 2007/10/18 10:38:24 chrjones Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Common/interface/RefCore.h"

namespace edm {

  template<typename KEY>
  class RefBase {
  public:
    typedef KEY key_type;

    /// Default constructor needed for reading from persistent store. Not for direct use.
    RefBase() : product_(), item_() {}

    /// General purpose constructor. 
    RefBase(ProductID const& productID, void const* prodPtr, key_type itemKey,
            void const* itemPtr, EDProductGetter const* prodGetter, bool transient):
      product_(productID, prodPtr, prodGetter, transient), item_(itemKey, itemPtr) {}

    /// Constructor from RefVector. 
    RefBase(RefCore const& prod, RefItem<KEY> const& itm) :
      product_(prod), item_(itm) {}

    /// Compiler-generated copy constructor, assignment operator, and
    /// destructor do the right thing.

    /// Accessor for product ID and product getter.
    RefCore const& refCore() const { return product_;}

    /// Accessor for index and pointer
    RefItem<KEY> const& item() const {return item_;}

    // /// Return the index for the referenced element.
    // key_type key() const { return item_.key(); }
    

    /// Return true if this RefBase is non-null
    bool isValid() const { return item_.isValid(); }
    bool isNull() const { return item_.isNull(); }
    bool isNonnull() const { return item_.isNonnull(); }

  private:
    RefCore product_;
    RefItem<KEY> item_;
  };

  template <typename KEY>
  bool
  operator==(RefBase<KEY> const& lhs, RefBase<KEY> const& rhs) {
    return lhs.refCore() == rhs.refCore() && lhs.item() == rhs.item();
  }
  
  template <typename KEY>
  bool
  operator!=(RefBase<KEY> const& lhs, RefBase<KEY> const& rhs) {
    return !(lhs == rhs);
  }

  template <typename KEY>
  bool
  operator<(RefBase<KEY> const& lhs, RefBase<KEY> const& rhs) {
    return (lhs.refCore() == rhs.refCore() ?  lhs.item() < rhs.item() : lhs.refCore() < rhs.refCore());
  }
}
  
#endif
