#ifndef FWCore_Framework_BranchIDListHelper_h
#define FWCore_Framework_BranchIDListHelper_h

#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include <map>

namespace edm {
    
  class BranchIDListHelper {
  public:
    typedef std::pair<BranchListIndex, ProductIndex> IndexPair;
    typedef std::multimap<BranchID, IndexPair> BranchIDToIndexMap;
    typedef std::map<BranchListIndex, BranchListIndex> BranchListIndexMapper;
    BranchIDListHelper();
    static bool updateFromInput(BranchIDLists const& bidlists, std::string const& fileName);
    static void updateRegistries(ProductRegistry const& reg);
    static void fixBranchListIndexes(BranchListIndexes& indexes);
    static void clearRegistries();  // Use only for tests

    BranchListIndex producedBranchListIndex() const {return producedBranchListIndex_;}
    BranchIDToIndexMap const& branchIDToIndexMap() const {return branchIDToIndexMap_;}

  private:
    BranchListIndex producedBranchListIndex_;
    BranchIDToIndexMap branchIDToIndexMap_;
    BranchListIndexMapper branchListIndexMapper_;
  };
}

#endif
