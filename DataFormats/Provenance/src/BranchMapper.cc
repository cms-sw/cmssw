#include "DataFormats/Provenance/interface/BranchMapper.h"

/*
  BranchMapper

*/

namespace edm {
  BranchMapper::BranchMapper() :
    entryInfoSet_(),
    entryInfoMap_(),
    nextMapper_(),
    delayedRead_(false)
  { }

  BranchMapper::BranchMapper(bool delayedRead) :
    entryInfoSet_(),
    entryInfoMap_(),
    nextMapper_(),
    delayedRead_(delayedRead)
  { }

  void
  BranchMapper::readProvenance() const {
    if (delayedRead_) {
      delayedRead_ = false;
      readProvenance_();
    }
  }

  void
  BranchMapper::insert(EventEntryInfo const& entryInfo) {
    readProvenance();
    entryInfoSet_.insert(entryInfo);
    if (!entryInfoMap_.empty()) {
      entryInfoMap_.insert(std::make_pair(entryInfo.productID(), entryInfoSet_.find(entryInfo)));
    }
  }
    
  boost::shared_ptr<EventEntryInfo>
  BranchMapper::branchToEntryInfo(BranchID const& bid) const {
    readProvenance();
    EventEntryInfo ei(bid);
    eiSet::const_iterator it = entryInfoSet_.find(ei);
    if (it == entryInfoSet_.end()) {
      if (nextMapper_) {
	return nextMapper_->branchToEntryInfo(bid);
      } else {
	return boost::shared_ptr<EventEntryInfo>();
      }
    }
    return boost::shared_ptr<EventEntryInfo>(new EventEntryInfo(*it));
  }

/*
  ProductID 
  BranchMapper::branchToProduct(BranchID const& bid) const {
    readProvenance();
    EventEntryInfo ei(bid);
    typename eiSet::const_iterator it = entryInfoSet_.find(ei);
    if (it == entryInfoSet_.end()) {
      if (nextMapper_) {
	return nextMapper_->branchToProduct(bid);
      } else {
	return ProductID();
      }
    }
    return it->productID();
  }
*/

  BranchID 
  BranchMapper::productToBranch(ProductID const& pid) const {
    readProvenance();
    if (entryInfoMap_.empty()) {
      eiMap & map = const_cast<eiMap &>(entryInfoMap_);
      for (eiSet::const_iterator i = entryInfoSet_.begin(), iEnd = entryInfoSet_.end();
	  i != iEnd; ++i) {
	map.insert(std::make_pair(i->productID(), i));
      }
    }
    eiMap::const_iterator it = entryInfoMap_.find(pid);
    if (it == entryInfoMap_.end()) {
      if (nextMapper_) {
	return nextMapper_->productToBranch(pid);
      } else {
	return BranchID();
      }
    }
    return it->second->branchID();
  }

  bool
  BranchMapper::fpred(eiSet::value_type const& a, eiSet::value_type const& b) {
    return a.productID() < b.productID();
  }

  ProductID
  BranchMapper::maxProductID() const {
    readProvenance();
    if (entryInfoSet_.empty()) {
      return ProductID(0);
    } else {
      eiSet::const_iterator it = std::max_element(entryInfoSet_.begin(), entryInfoSet_.end(), fpred);
      return it->productID();
    }
  }
}
