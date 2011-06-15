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
  class ProductID;
  class BranchMapper {
  public:
    BranchMapper();

    explicit BranchMapper(bool delayedRead);

    virtual ~BranchMapper();

    void write(std::ostream& os) const;

    ProductProvenance const* branchIDToProvenance(BranchID const& bid) const;

    void insert(ProductProvenance const& provenanceProduct);

    void mergeMappers(boost::shared_ptr<BranchMapper> other) {nextMapper_ = other;}

    void setDelayedRead(bool value) {delayedRead_ = value;}

    BranchID oldProductIDToBranchID(ProductID const& oldProductID) const {
      return oldProductIDToBranchID_(oldProductID);
    }

    ProcessHistoryID const& processHistoryID() const {return processHistoryID_;}

    ProcessHistoryID& processHistoryID() {return processHistoryID_;}

    void reset();
  private:
    typedef std::set<ProductProvenance> eiSet;

    void readProvenance() const;
    virtual void readProvenance_() const {}
    virtual void reset_() {}
    
    virtual BranchID oldProductIDToBranchID_(ProductID const& oldProductID) const;

    eiSet entryInfoSet_;

    boost::shared_ptr<BranchMapper> nextMapper_;

    mutable bool delayedRead_;

    ProcessHistoryID processHistoryID_;
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, BranchMapper const& p) {
    p.write(os);
    return os;
  }
}
#endif
