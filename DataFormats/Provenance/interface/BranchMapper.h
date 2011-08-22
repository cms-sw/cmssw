#ifndef DataFormats_Provenance_BranchMapper_h
#define DataFormats_Provenance_BranchMapper_h

/*----------------------------------------------------------------------
  
BranchMapper: Manages the per event/lumi/run per product provenance.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/utility.hpp"

#include <iosfwd>
#include <memory>
#include <set>

/*
  BranchMapper
*/

namespace edm {
  class ProvenanceReaderBase;

  class BranchMapper : private boost::noncopyable {
  public:
    BranchMapper();
    explicit BranchMapper(std::auto_ptr<ProvenanceReaderBase> reader);

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
    mutable boost::scoped_ptr<ProvenanceReaderBase> provenanceReader_;
  };

  class ProvenanceReaderBase {
  public:
    ProvenanceReaderBase() {}
    virtual ~ProvenanceReaderBase();
    virtual void readProvenance(BranchMapper const& mapper) const = 0;
  };
  
}
#endif
