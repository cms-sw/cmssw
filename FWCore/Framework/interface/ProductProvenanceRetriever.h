#ifndef DataFormats_Provenance_ProductProvenanceRetriever_h
#define DataFormats_Provenance_ProductProvenanceRetriever_h

/*----------------------------------------------------------------------
  
ProductProvenanceRetriever: Manages the per event/lumi/run per product provenance.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProductProvenanceLookup.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <memory>
#include <set>
#include <atomic>

/*
  ProductProvenanceRetriever
*/

namespace edm {
  class ModuleCallingContext;
  class ProductRegistry;

  class ProvenanceReaderBase {
  public:
    ProvenanceReaderBase() {}
    virtual ~ProvenanceReaderBase();
    virtual std::set<ProductProvenance> readProvenance(unsigned int transitionIndex) const = 0;
    virtual void readProvenanceAsync(WaitingTaskHolder task,
                                     ModuleCallingContext const* moduleCallingContext,
                                     unsigned int transitionIndex,
                                     std::atomic<const std::set<ProductProvenance>*>& writeTo) const noexcept = 0;

    virtual void unsafe_fillProvenance(unsigned int transitionIndex) const;
  };

  class ProductProvenanceRetriever : public ProductProvenanceLookup {
  public:
    explicit ProductProvenanceRetriever(unsigned int iTransitionIndex);
    ProductProvenanceRetriever(unsigned int iTransitionIndex, edm::ProductRegistry const&);
    explicit ProductProvenanceRetriever(unsigned int iTransitionIndex, std::unique_ptr<ProvenanceReaderBase> reader);

    ProductProvenanceRetriever& operator=(ProductProvenanceRetriever const&) = delete;

    void mergeProvenanceRetrievers(std::shared_ptr<ProductProvenanceRetriever> other);

    void mergeParentProcessRetriever(ProductProvenanceRetriever const& provRetriever);

    void deepCopy(ProductProvenanceRetriever const&);

    void reset();

    void readProvenanceAsync(WaitingTaskHolder task, ModuleCallingContext const* moduleCallingContext) const noexcept;

    // Used in prompt reading mode to fill the branch at the same time
    // when all event data is read
    void unsafe_fillProvenance();

  private:
    std::unique_ptr<const std::set<ProductProvenance>> readProvenance() const final;
    const ProductProvenanceLookup* nextRetriever() const final { return nextRetriever_.get(); }

    edm::propagate_const<std::shared_ptr<ProductProvenanceRetriever>> nextRetriever_;
    std::shared_ptr<const ProvenanceReaderBase> provenanceReader_;
    unsigned int transitionIndex_;
  };

}  // namespace edm
#endif
