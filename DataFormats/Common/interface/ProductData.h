#ifndef DataFormats_Common_ProductData_h
#define DataFormats_Common_ProductData_h

/*----------------------------------------------------------------------

ProductData: A collection of information related to a single EDProduct. This
is the storage unit of such information.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/Provenance.h"
#include "boost/shared_ptr.hpp"

namespace edm {
  class ConstBranchDescription;
  class WrapperOwningHolder;
  struct ProductData {
    ProductData();

    explicit ProductData(boost::shared_ptr<ConstBranchDescription> bd);

    // For use by FWLite
    ProductData(void const* product, Provenance const& prov);

    WrapperInterfaceBase const* getInterface() const {
      return prov_.product().getInterface();
    }

    boost::shared_ptr<ConstBranchDescription> const& branchDescription() const {
      return prov_.constBranchDescriptionPtr();
    }

    void swap(ProductData& other) {
       std::swap(wrapper_, other.wrapper_);
       prov_.swap(other.prov_);
    }

    void resetBranchDescription(boost::shared_ptr<ConstBranchDescription> bd);

    void resetProductData() {
      wrapper_.reset();
      prov_.resetProductProvenance();
    }

    // NOTE: We should probably think hard about whether these
    // variables should be declared "mutable" as part of
    // the effort to make the Framework multithread capable ...

    // "non-const data" (updated every event)
    mutable boost::shared_ptr<void const> wrapper_;
    mutable Provenance prov_;
  };

  // Free swap function
  inline void swap(ProductData& a, ProductData& b) {
    a.swap(b);
  }
}
#endif
