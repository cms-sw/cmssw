#include "DataFormats/Provenance/interface/BranchIDListHelper.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include <cassert>

namespace edm {

  BranchIDListHelper::BranchIDListHelper() :
    branchIDLists_(),
    branchIDToIndexMap_(),
    inputIndexToJobIndex_(),
    producedBranchListIndex_(std::numeric_limits<BranchListIndex>::max()),
    nAlreadyCopied_(0)
  {}

  bool
  BranchIDListHelper::updateFromInput(BranchIDLists const& bidlists) {
    //The BranchIDLists is a list of lists
    // this routine compares bidlists to branchIDLists_ to see if a list
    // in branchIDLists_ is already in bidlist and if it isn't we insert
    // that new list into branchIDLists_
    bool unchanged = true;
    inputIndexToJobIndex_.clear();
    inputIndexToJobIndex_.resize(bidlists.size());
    for(auto it = bidlists.begin(), itEnd = bidlists.end(); it != itEnd; ++it) {
      BranchListIndex oldBlix = it - bidlists.begin();
      auto j = find_in_all(branchIDLists_, *it);
      BranchListIndex blix = j - branchIDLists_.begin();
      if(j == branchIDLists_.end()) {
        branchIDLists_.push_back(*it);
        for(BranchIDList::const_iterator i = it->begin(), iEnd = it->end(); i != iEnd; ++i) {
          ProductIndex pix = i - it->begin();
          branchIDToIndexMap_.insert(std::make_pair(BranchID(*i), std::make_pair(blix, pix)));
        }
      }
      inputIndexToJobIndex_[oldBlix]=blix;
      if(oldBlix != blix) {
        unchanged = false;
      }
    }
    return unchanged;
  }

  void
  BranchIDListHelper::updateFromParent(BranchIDLists const& bidlists) {

    inputIndexToJobIndex_.resize(bidlists.size());
    for(auto it = bidlists.begin() + nAlreadyCopied_, itEnd = bidlists.end(); it != itEnd; ++it) {
      BranchListIndex oldBlix = it - bidlists.begin();
      BranchListIndex blix = branchIDLists_.size();
      branchIDLists_.push_back(*it);
      for(BranchIDList::const_iterator i = it->begin(), iEnd = it->end(); i != iEnd; ++i) {
        ProductIndex pix = i - it->begin();
        branchIDToIndexMap_.insert(std::make_pair(BranchID(*i), std::make_pair(blix, pix)));
      }
      inputIndexToJobIndex_[oldBlix]=blix;
    }
    nAlreadyCopied_ = bidlists.size();
  }

  void
  BranchIDListHelper::updateFromRegistry(ProductRegistry const& preg) {
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
      producedBranchListIndex_ = blix;
      //preg.setProducedBranchListIndex(blix);
      branchIDLists_.push_back(bidlist);
      for(BranchIDList::const_iterator i = bidlist.begin(), iEnd = bidlist.end(); i != iEnd; ++i) {
        ProductIndex pix = i - bidlist.begin();
        branchIDToIndexMap_.insert(std::make_pair(BranchID(*i), std::make_pair(blix, pix)));
      }
    }
  }

  void
  BranchIDListHelper::fixBranchListIndexes(BranchListIndexes& indexes) const {
    for(BranchListIndex& i : indexes) {
      assert(i<inputIndexToJobIndex_.size());
      i = inputIndexToJobIndex_[i];
    }
  }
}
