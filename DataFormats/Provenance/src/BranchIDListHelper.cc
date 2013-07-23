#include "DataFormats/Provenance/interface/BranchIDListHelper.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/Algorithms.h"

namespace edm {

  BranchIDListHelper::BranchIDListHelper() :
    branchIDLists_(),
    branchIDToIndexMap_(),
    branchListIndexMapper_() {}

  bool
  BranchIDListHelper:: updateFromInput(BranchIDLists const& bidlists) {
    bool unchanged = true;
    branchListIndexMapper_.clear();
    typedef BranchIDLists::const_iterator Iter;
    for(Iter it = bidlists.begin(), itEnd = bidlists.end(); it != itEnd; ++it) {
      BranchListIndex oldBlix = it - bidlists.begin();
      Iter j = find_in_all(branchIDLists_, *it);
      BranchListIndex blix = j - branchIDLists_.begin();
      if(j == branchIDLists_.end()) {
        branchIDLists_.push_back(*it);
        for(BranchIDList::const_iterator i = it->begin(), iEnd = it->end(); i != iEnd; ++i) {
          ProductIndex pix = i - it->begin();
          branchIDToIndexMap_.insert(std::make_pair(BranchID(*i), std::make_pair(blix, pix)));
        }
      }
      branchListIndexMapper_.insert(std::make_pair(oldBlix, blix));
      if(oldBlix != blix) {
        unchanged = false;
      }
    }
    return unchanged;
  }

  void
  BranchIDListHelper::updateRegistries(ProductRegistry& preg) {
    BranchIDList bidlist;
    // Add entries for current process for ProductID to BranchID mapping.
    for(ProductRegistry::ProductList::const_iterator it = preg.productList().begin(), itEnd = preg.productList().end();
        it != itEnd; ++it) {
      if(it->second.produced()) {
        if(it->second.branchType() == InEvent) {
          bidlist.push_back(it->second.branchID().id());
        }
      }
    }
    if(!bidlist.empty()) {
      BranchListIndex blix = branchIDLists_.size();
      preg.setProducedBranchListIndex(blix);
      branchIDLists_.push_back(bidlist);
      for(BranchIDList::const_iterator i = bidlist.begin(), iEnd = bidlist.end(); i != iEnd; ++i) {
        ProductIndex pix = i - bidlist.begin();
        branchIDToIndexMap_.insert(std::make_pair(BranchID(*i), std::make_pair(blix, pix)));
      }
    }
  }

  void
  BranchIDListHelper::fixBranchListIndexes(BranchListIndexes& indexes) {
    for(BranchListIndex& i : indexes) {
      i = branchListIndexMapper_[i];
    }
  }
}
