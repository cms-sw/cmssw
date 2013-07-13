#ifndef FWCore_Framework_BranchIDListHelper_h
#define FWCore_Framework_BranchIDListHelper_h

#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"

#include <map>

namespace edm {
    
  class BranchIDListHelper {
  public:
    typedef std::pair<BranchListIndex, ProductIndex> IndexPair;
    typedef std::multimap<BranchID, IndexPair> BranchIDToIndexMap;
    typedef std::map<BranchListIndex, BranchListIndex> BranchListIndexMapper;
    BranchIDListHelper();
    bool updateFromInput(BranchIDLists const& bidlists);
    void updateRegistries(ProductRegistry& reg);
    void fixBranchListIndexes(BranchListIndexes& indexes);

    BranchIDLists const& branchIDLists() const {return branchIDLists_;}
    BranchIDLists& branchIDLists() {return branchIDLists_;}
    BranchIDToIndexMap const& branchIDToIndexMap() const {return branchIDToIndexMap_;}

  private:
    BranchIDLists branchIDLists_;
    BranchIDToIndexMap branchIDToIndexMap_;
    BranchListIndexMapper branchListIndexMapper_;
  };
}

#endif
