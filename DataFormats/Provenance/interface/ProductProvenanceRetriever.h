#ifndef DataFormats_Provenance_BranchMapper_h
#define DataFormats_Provenance_BranchMapper_h

/*----------------------------------------------------------------------
  
ProductProvenanceRetriever: Manages the per event/lumi/run per product provenance.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <memory>
#include <set>
#include <atomic>

/*
  ProductProvenanceRetriever
*/

namespace edm {
  class ProvenanceReaderBase;

  class ProductProvenanceRetriever {
  public:
    explicit ProductProvenanceRetriever(unsigned int iTransitionIndex);
    explicit ProductProvenanceRetriever(std::unique_ptr<ProvenanceReaderBase> reader);
    
    ProductProvenanceRetriever& operator=(ProductProvenanceRetriever const&) = delete;

    ~ProductProvenanceRetriever();

    ProductProvenance const* branchIDToProvenance(BranchID const& bid) const;

    void insertIntoSet(ProductProvenance const& provenanceProduct) const;

    void mergeProvenanceRetrievers(std::shared_ptr<ProductProvenanceRetriever> other);

    void deepCopy(ProductProvenanceRetriever const&);
    
    void reset();
  private:
    void readProvenance() const;
    void setTransitionIndex(unsigned int transitionIndex) {
      transitionIndex_=transitionIndex;
    }

    mutable std::set<ProductProvenance> entryInfoSet_;
    mutable std::atomic<const std::set<ProductProvenance>*> readEntryInfoSet_;
    edm::propagate_const<std::shared_ptr<ProductProvenanceRetriever>> nextRetriever_;
    std::shared_ptr<const ProvenanceReaderBase> provenanceReader_;
    unsigned int transitionIndex_;
  };

  class ProvenanceReaderBase {
  public:
    ProvenanceReaderBase() {}
    virtual ~ProvenanceReaderBase();
    virtual std::set<ProductProvenance> readProvenance(unsigned int transitionIndex) const = 0;
  };
  
}
#endif
