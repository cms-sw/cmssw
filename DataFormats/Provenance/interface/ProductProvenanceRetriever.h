#ifndef DataFormats_Provenance_BranchMapper_h
#define DataFormats_Provenance_BranchMapper_h

/*----------------------------------------------------------------------
  
ProductProvenanceRetriever: Manages the per event/lumi/run per product provenance.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

#include "boost/scoped_ptr.hpp"
#include "boost/utility.hpp"

#include <iosfwd>
#include <memory>
#include <set>

/*
  ProductProvenanceRetriever
*/

namespace edm {
  class ProvenanceReaderBase;

  class ProductProvenanceRetriever : private boost::noncopyable {
  public:
    explicit ProductProvenanceRetriever(unsigned int iTransitionIndex);
#ifndef __GCCXML__
    explicit ProductProvenanceRetriever(std::unique_ptr<ProvenanceReaderBase> reader);
#endif

    ~ProductProvenanceRetriever();

    ProductProvenance const* branchIDToProvenance(BranchID const& bid) const;

    void insertIntoSet(ProductProvenance const& provenanceProduct) const;

    void mergeProvenanceRetrievers(std::shared_ptr<ProductProvenanceRetriever> other);

    void deepSwap(ProductProvenanceRetriever&);
    
    void reset();
  private:
    void readProvenance() const;
    void setTransitionIndex(unsigned int transitionIndex) {
      transitionIndex_=transitionIndex;
    }

    typedef std::set<ProductProvenance> eiSet;

    mutable eiSet entryInfoSet_;
    std::shared_ptr<ProductProvenanceRetriever> nextRetriever_;
    mutable std::shared_ptr<ProvenanceReaderBase> provenanceReader_;
    unsigned int transitionIndex_;
    mutable bool delayedRead_;
  };

  class ProvenanceReaderBase {
  public:
    ProvenanceReaderBase() {}
    virtual ~ProvenanceReaderBase();
    virtual void readProvenance(ProductProvenanceRetriever const& mapper, unsigned int transitionIndex) const = 0;
  };
  
}
#endif
