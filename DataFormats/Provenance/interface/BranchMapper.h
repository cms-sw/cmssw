#ifndef DataFormats_Provenance_BranchMapper_h
#define DataFormats_Provenance_BranchMapper_h

/*----------------------------------------------------------------------
  
BranchMapper: The mapping from per event product ID's to BranchID's.

----------------------------------------------------------------------*/
#include <iosfwd>
#include <set>
#include <map>
#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "FWCore/Utilities/interface/Algorithms.h"

/*
  BranchMapper

*/

namespace edm {
  class BranchMapper {
  public:
    BranchMapper();

    explicit BranchMapper(bool delayedRead);

    virtual ~BranchMapper() {}

    void write(std::ostream& os) const;

    BranchID productToBranch(ProductID const& pid) const;
    
    boost::shared_ptr<EventEntryInfo> branchToEntryInfo(BranchID const& bid) const;

    void insert(EventEntryInfo const& entryInfo);

    void mergeMappers(boost::shared_ptr<BranchMapper> other) {nextMapper_ = other;}

    ProductID maxProductID() const;

  private:
    typedef std::set<EventEntryInfo> eiSet;
    typedef std::map<ProductID, eiSet::const_iterator> eiMap;
    static bool fpred(eiSet::value_type const& a, eiSet::value_type const& b);

    void readProvenance() const;
    virtual void readProvenance_() const {}

    eiSet entryInfoSet_;

    eiMap entryInfoMap_;

    boost::shared_ptr<BranchMapper> nextMapper_;

    mutable bool delayedRead_;

  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, BranchMapper const& p) {
    p.write(os);
    return os;
  }
}
#endif
