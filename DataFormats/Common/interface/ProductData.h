#ifndef DataFormats_Common_ProductData_h
#define DataFormats_Common_ProductData_h

/*----------------------------------------------------------------------

ProductData: A collection of information related to a single EDProduct. This
is the storage unit of such information.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/Provenance.h"
#include <memory>

namespace edm {
  class BranchDescription;
  class WrapperBase;
  struct ProductData {
    ProductData();

    explicit ProductData(std::shared_ptr<BranchDescription const> bd);

    // For use by FWLite
    ProductData(WrapperBase* product, Provenance const& prov);

    std::shared_ptr<BranchDescription const> const& branchDescription() const {
      return prov_.constBranchDescriptionPtr();
    }

    void swap(ProductData& other) {
       std::swap(wrapper_, other.wrapper_);
       prov_.swap(other.prov_);
    }

    void resetBranchDescription(std::shared_ptr<BranchDescription const> bd);

    void resetProductData() {
      wrapper_.reset();
      prov_.resetProductProvenance();
    }

    // NOTE: We should probably think hard about whether these
    // variables should be declared "mutable" as part of
    // the effort to make the Framework multithread capable ...

    // "non-const data" (updated every event)
    mutable std::shared_ptr<WrapperBase> wrapper_;
    mutable Provenance prov_;
  };

  // Free swap function
  inline void swap(ProductData& a, ProductData& b) {
    a.swap(b);
  }
}
#endif
