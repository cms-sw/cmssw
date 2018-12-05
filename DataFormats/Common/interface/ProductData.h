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
  class MergeableRunProductMetadataBase;
  class WrapperBase;

  class ProductData {
  public:
    ProductData();

    explicit ProductData(std::shared_ptr<BranchDescription const> bd);

    // For use by FWLite
    ProductData(WrapperBase* product, Provenance const& prov);

    std::shared_ptr<BranchDescription const> const& branchDescription() const {
      return prov_.constBranchDescriptionPtr();
    }
    
    Provenance const& provenance() const { return prov_;}
    
    WrapperBase const* wrapper() const { return wrapper_.get();}
    WrapperBase* wrapper() { return wrapper_.get(); }
    WrapperBase* unsafe_wrapper() const { return wrapper_.get(); }
    std::shared_ptr<WrapperBase const> sharedConstWrapper() const {
      return wrapper_;
    }
    
    void swap(ProductData& other) {
       std::swap(wrapper_, other.wrapper_);
       prov_.swap(other.prov_);
    }

    void setWrapper(std::unique_ptr<WrapperBase> iValue);
    
    //Not const thread-safe update
    void unsafe_setWrapper(std::unique_ptr<WrapperBase> iValue) const;
    void unsafe_setWrapper(std::shared_ptr<WrapperBase const> iValue) const; // for SwitchProducer
    
    void resetBranchDescription(std::shared_ptr<BranchDescription const> bd);

    void resetProductData() {
      wrapper_.reset();
    }

    void unsafe_resetProductData() const {
      wrapper_.reset();
    }

    void setProcessHistory(ProcessHistory const& ph) {
      prov_.setProcessHistory(ph);
    }
    
    void setProvenance(ProductProvenanceRetriever const* provRetriever, ProcessHistory const& ph, ProductID const& pid) {
      prov_.setProductID(pid);
      prov_.setStore(provRetriever);
      prov_.setProcessHistory(ph);
    }

    void setMergeableRunProductMetadata(MergeableRunProductMetadataBase const* mrpm) {
      prov_.setMergeableRunProductMetadata(mrpm);
    }

    // NOTE: We should probably think hard about whether these
    // variables should be declared "mutable" as part of
    // the effort to make the Framework multithread capable ...

  private:
    // "non-const data" (updated every event)
    mutable std::shared_ptr<WrapperBase> wrapper_;
    Provenance prov_;
  };

  // Free swap function
  inline void swap(ProductData& a, ProductData& b) {
    a.swap(b);
  }
}
#endif
