#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/BranchIDListRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include <limits>
#include <iostream>

namespace edm {

  
  BranchIDListHelper::BranchIDListHelper() :
    producedBranchListIndex_(std::numeric_limits<BranchListIndex>::max()),
    branchIDToIndexMap_(),
    branchListIndexMapper_() {}

  void
  BranchIDListHelper:: updateFromInput(BranchIDLists const& bidlists, std::string const& fileName) {
    BranchIDListRegistry& breg = *BranchIDListRegistry::instance();
    BranchIDListRegistry::collection_type& bdata = breg.data();
    BranchIDToIndexMap& branchIDToIndexMap = breg.extra().branchIDToIndexMap_;
    BranchListIndexMapper& branchListIndexMapper = breg.extra().branchListIndexMapper_;
    branchListIndexMapper.clear();
    typedef BranchIDLists::const_iterator Iter;
    typedef BranchIDListRegistry::const_iterator RegIter;
    for (Iter it = bidlists.begin(), itEnd = bidlists.end(); it != itEnd; ++it) {
      BranchListIndex oldBlix = it - bidlists.begin();
      RegIter j = find_in_all(bdata, *it);
      BranchListIndex blix = j - bdata.begin();
      if (j == bdata.end()) {
	breg.insertMapped(*it);
	for (BranchIDList::const_iterator i = it->begin(), iEnd = it->end(); i != iEnd; ++i) {
	  ProductIndex pix = i - it->begin();
	  branchIDToIndexMap.insert(std::make_pair(*i, std::make_pair(blix, pix)));
	}
      }
      branchListIndexMapper.insert(std::make_pair(oldBlix, blix));
    }
    BranchListIndex producedBranchListIndex = breg.extra().producedBranchListIndex_;
    if (producedBranchListIndex != std::numeric_limits<BranchListIndex>::max()) {
      branchListIndexMapper.insert(std::make_pair(bidlists.size(), producedBranchListIndex));
    }
  }

  void
  BranchIDListHelper::updateRegistries(ProductRegistry const& preg) {
    BranchIDList bidlist;
    // Add entries for current process for ProductID to BranchID mapping.
    for (ProductRegistry::ProductList::const_iterator it = preg.productList().begin(), itEnd = preg.productList().end();
        it != itEnd; ++it) {
      if (it->second.produced()) {
        if (it->second.branchType() == InEvent) {
          bidlist.push_back(it->second.branchID().id());
        }
      }
    }
    BranchIDListRegistry& breg = *BranchIDListRegistry::instance();
    BranchIDToIndexMap& branchIDToIndexMap = breg.extra().branchIDToIndexMap_;
    BranchListIndexMapper& branchListIndexMapper = breg.extra().branchListIndexMapper_;
    if (!bidlist.empty()) {
      BranchListIndex blix = breg.data().size();
      breg.extra().producedBranchListIndex_ = blix;
      breg.insertMapped(bidlist);
      for (BranchIDList::const_iterator i = bidlist.begin(), iEnd = bidlist.end(); i != iEnd; ++i) {
        ProductIndex pix = i - bidlist.begin();
	branchIDToIndexMap.insert(std::make_pair(*i, std::make_pair(blix, pix)));
      }
      branchListIndexMapper.insert(std::make_pair(blix, blix));
    }
  }

  void
  BranchIDListHelper::fixBranchListIndexes(BranchListIndexes& indexes) {
    BranchIDListRegistry& breg = *BranchIDListRegistry::instance();
    BranchListIndexMapper& branchListIndexMapper = breg.extra().branchListIndexMapper_;
    for (BranchListIndexes::iterator i = indexes.begin(), e = indexes.end(); i != e; ++i) {
      *i = branchListIndexMapper[i - indexes.begin()];
    }
  }

  void
  BranchIDListHelper::clearRegistries() {
    BranchIDListRegistry& breg = *BranchIDListRegistry::instance();
    breg.data().clear();
    breg.extra().producedBranchListIndex_ = std::numeric_limits<BranchListIndex>::max();
    breg.extra().branchIDToIndexMap_.clear();
    breg.extra().branchListIndexMapper_.clear();
  }
}
