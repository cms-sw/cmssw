#ifndef DataFormats_Provenance_BranchMapper_h
#define DataFormats_Provenance_BranchMapper_h

/*----------------------------------------------------------------------
  
ProductProvenanceRetriever: Manages the per event/lumi/run per product provenance.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include "tbb/concurrent_unordered_set.h"
#include <memory>
#include <set>
#include <atomic>

/*
  ProductProvenanceRetriever
*/

namespace edm {
  class ProvenanceReaderBase;
  class WaitingTask;
  class ModuleCallingContext;

  struct ProductProvenanceHasher {
    size_t operator()(ProductProvenance const& tid) const { return tid.branchID().id(); }
  };

  struct ProductProvenanceEqual {
    //The default operator== for ProductProvenance does not work for this case
    // Since (a<b)==false  and (a>b)==false does not mean a==b for the operators
    // defined for ProductProvenance
    bool operator()(ProductProvenance const& iLHS, ProductProvenance const iRHS) const {
      return iLHS.branchID().id() == iRHS.branchID().id();
    }
  };

  class ProvenanceReaderBase {
  public:
    ProvenanceReaderBase() {}
    virtual ~ProvenanceReaderBase();
    virtual std::set<ProductProvenance> readProvenance(unsigned int transitionIndex) const = 0;
    virtual void readProvenanceAsync(WaitingTask* task,
                                     ModuleCallingContext const* moduleCallingContext,
                                     unsigned int transitionIndex,
                                     std::atomic<const std::set<ProductProvenance>*>& writeTo) const = 0;
  };

  class ProductProvenanceRetriever {
  public:
    explicit ProductProvenanceRetriever(unsigned int iTransitionIndex);
    explicit ProductProvenanceRetriever(std::unique_ptr<ProvenanceReaderBase> reader);

    ProductProvenanceRetriever& operator=(ProductProvenanceRetriever const&) = delete;

    ~ProductProvenanceRetriever();

    ProductProvenance const* branchIDToProvenance(BranchID const& bid) const;

    ProductProvenance const* branchIDToProvenanceForProducedOnly(BranchID const& bid) const;

    void insertIntoSet(ProductProvenance provenanceProduct) const;

    void mergeProvenanceRetrievers(std::shared_ptr<ProductProvenanceRetriever> other);

    void mergeParentProcessRetriever(ProductProvenanceRetriever const& provRetriever);

    void deepCopy(ProductProvenanceRetriever const&);

    void reset();

    void readProvenanceAsync(WaitingTask* task, ModuleCallingContext const* moduleCallingContext) const;

  private:
    void readProvenance() const;
    void setTransitionIndex(unsigned int transitionIndex) { transitionIndex_ = transitionIndex; }

    mutable tbb::concurrent_unordered_set<ProductProvenance, ProductProvenanceHasher, ProductProvenanceEqual>
        entryInfoSet_;
    mutable std::atomic<const std::set<ProductProvenance>*> readEntryInfoSet_;
    edm::propagate_const<std::shared_ptr<ProductProvenanceRetriever>> nextRetriever_;
    edm::propagate_const<ProductProvenanceRetriever const*> parentProcessRetriever_;
    std::shared_ptr<const ProvenanceReaderBase> provenanceReader_;
    unsigned int transitionIndex_;
  };

}  // namespace edm
#endif
