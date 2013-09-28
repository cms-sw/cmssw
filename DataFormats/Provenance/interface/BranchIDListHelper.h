#ifndef FWCore_Framework_BranchIDListHelper_h
#define FWCore_Framework_BranchIDListHelper_h

#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"

#include <map>
#include <vector>

namespace edm {
    
  class BranchIDListHelper {
  public:
    typedef std::pair<BranchListIndex, ProductIndex> IndexPair;
    typedef std::multimap<BranchID, IndexPair> BranchIDToIndexMap;

    BranchIDListHelper();

    //CMS-THREADING called when a new file is opened
    bool updateFromInput(BranchIDLists const& bidlists);
    ///Called by sources to convert their read indexes into the indexes used by the job
    void fixBranchListIndexes(BranchListIndexes& indexes) const;

    void updateRegistries(ProductRegistry& reg);

    //CMS-THREADING this is called in SubJob::beginJob
    BranchIDLists& mutableBranchIDLists() {return branchIDLists_;}

    //Used by the EventPrincipal
    BranchIDLists const& branchIDLists() const {return branchIDLists_;}
    BranchIDToIndexMap const& branchIDToIndexMap() const {return branchIDToIndexMap_;}

  private:
    BranchIDLists branchIDLists_;
    BranchIDToIndexMap branchIDToIndexMap_;
    std::vector<BranchListIndex> inputIndexToJobIndex_;
  };
}

#endif
