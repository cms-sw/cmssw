#ifndef DataFormats_Common_interface_OrphanHandleBase_h
#define DataFormats_Common_interface_OrphanHandleBase_h

/*----------------------------------------------------------------------
  
OrphanHandle: Non-owning "smart pointer" for reference to EDProducts.

This is a very preliminary version, and lacks safety features and
elegance.

If the pointed-to WrapperBase is destroyed, use of the OrphanHandle
becomes undefined. There is no way to query the OrphanHandle to
discover if this has happened.

OrphanHandles can have:
  -- Product pointer null and id == 0;
  -- Product pointer valid and id != 0;

To check validity, one can use the isValid() function.

----------------------------------------------------------------------*/

#include <algorithm>
#include <cassert>

#include "DataFormats/Provenance/interface/ProductID.h"

namespace edm {

  class OrphanHandleBase {
  public:
    OrphanHandleBase() : product_(nullptr), id_() {}

    OrphanHandleBase(void const* iProd, ProductID const& iId) : product_(iProd), id_(iId) { assert(iProd); }

    void clear() {
      product_ = nullptr;
      id_ = ProductID();
    }

    void swap(OrphanHandleBase& other) {
      std::swap(product_, other.product_);
      std::swap(id_, other.id_);
    }

    bool isValid() const { return product_ && id_ != ProductID(); }

    ProductID id() const { return id_; }

  protected:
    void const* productStorage() const { return product_; }

  private:
    void const* product_;
    ProductID id_;
  };

  // Free swap function
  inline void swap(OrphanHandleBase& a, OrphanHandleBase& b) { a.swap(b); }

}  // namespace edm

#endif  // DataFormats_Common_interface_OrphanHandleBase_h
