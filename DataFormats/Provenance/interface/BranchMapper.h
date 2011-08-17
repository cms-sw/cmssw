#ifndef DataFormats_Provenance_BranchMapper_h
#define DataFormats_Provenance_BranchMapper_h

/*----------------------------------------------------------------------
  
BranchMapper: Manages the per event/lumi/run per product provenance.

----------------------------------------------------------------------*/
#include <iosfwd>
#include <set>
#include <map>

#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

/*
  BranchMapper

*/

namespace edm {
  class ProvenanceReaderBase;

  class BranchMapper {
  public:
    BranchMapper();
    explicit BranchMapper(boost::shared_ptr<ProvenanceReaderBase> reader);

    ~BranchMapper();

    ProductProvenance const* branchIDToProvenance(BranchID const& bid) const;

    void insertIntoSet(ProductProvenance const& provenanceProduct) const;

    void mergeMappers(boost::shared_ptr<BranchMapper> other);

    void reset();
  private:
    void readProvenance() const;

    typedef std::set<ProductProvenance> eiSet;

    mutable eiSet entryInfoSet_;
    boost::shared_ptr<BranchMapper> nextMapper_;
    mutable bool delayedRead_;
    mutable boost::shared_ptr<ProvenanceReaderBase> provenanceReader_;
  };

  class ProvenanceReaderBase {
  public:
    ProvenanceReaderBase() {}
    virtual ~ProvenanceReaderBase();
    virtual void readProvenance(BranchMapper const& mapper) const = 0;
  };
  
}
#endif
