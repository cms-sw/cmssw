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
    typedef std::map<BranchID, IndexPair> BranchIDToIndexMap;
    BranchIDListHelper() : branchIDToIndexMap_() {}
    static void updateFromInput(BranchIDLists const& bidlists, std::string const& fileName);
    static void updateRegistries(ProductRegistry const& reg);
    static void clearRegistries();  // Use only for tests

    BranchIDToIndexMap const& branchIDToIndexMap() const {return branchIDToIndexMap_;}

  private:
    BranchIDToIndexMap branchIDToIndexMap_;
  };
}

#endif
