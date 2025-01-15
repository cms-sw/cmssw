#ifndef FWCore_Framework_DroppedDataProductResolver_h
#define FWCore_Framework_DroppedDataProductResolver_h

/*----------------------------------------------------------------------

ProductResolver: Class to handle access to a WrapperBase and its related information.

 [The class was formerly called Group and later ProductHolder]
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProductResolverBase.h"

namespace edm {
  class DroppedDataProductResolver : public ProductResolverBase {
  public:
    DroppedDataProductResolver(std::shared_ptr<BranchDescription const> bd)
        : ProductResolverBase(), m_provenance(bd, {}) {}

    void connectTo(ProductResolverBase const&, Principal const*) final {}

  private:
    Resolution resolveProduct_(Principal const& principal,
                               bool skipCurrentProcess,
                               SharedResourcesAcquirer* sra,
                               ModuleCallingContext const* mcc) const final;
    void prefetchAsync_(WaitingTaskHolder waitTask,
                        Principal const& principal,
                        bool skipCurrentProcess,
                        ServiceToken const& token,
                        SharedResourcesAcquirer* sra,
                        ModuleCallingContext const* mcc) const noexcept final;

    void retrieveAndMerge_(Principal const& principal,
                           MergeableRunProductMetadata const* mergeableRunProductMetadata) const final;
    bool productUnavailable_() const final { return true; }
    bool productResolved_() const final { return true; }
    bool productWasDeleted_() const final { return false; }
    bool productWasFetchedAndIsValid_(bool iSkipCurrentProcess) const final { return false; }
    bool unscheduledWasNotRun_() const final { return false; }
    void resetProductData_(bool deleteEarly) final {}
    BranchDescription const& branchDescription_() const final { return m_provenance.branchDescription(); }
    void resetBranchDescription_(std::shared_ptr<BranchDescription const> bd) final {
      m_provenance.setBranchDescription(bd);
    }
    Provenance const* provenance_() const final { return &m_provenance; }

    std::string const& resolvedModuleLabel_() const final { return moduleLabel(); }
    void setProductProvenanceRetriever_(ProductProvenanceRetriever const* provRetriever) final;
    void setProductID_(ProductID const& pid) final { m_provenance.setProductID(pid); }
    ProductProvenance const* productProvenancePtr_() const final { return m_provenance.productProvenance(); }
    bool singleProduct_() const final { return true; }

    Provenance m_provenance;
  };
}  // namespace edm

#endif
