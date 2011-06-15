#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "FWCore/Utilities/interface/EDMException.h"

/*
  BranchMapper

*/

namespace edm {
  BranchMapper::BranchMapper() :
    entryInfoSet_(),
    nextMapper_(),
    delayedRead_(false),
    processHistoryID_()
  { }

  BranchMapper::BranchMapper(bool delayedRead) :
    entryInfoSet_(),
    nextMapper_(),
    delayedRead_(delayedRead),
    processHistoryID_()
  { }

  BranchMapper::~BranchMapper() {}

  void
  BranchMapper::readProvenance() const {
    if (delayedRead_) {
      delayedRead_ = false;
      readProvenance_();
    }
  }

  void
  BranchMapper::reset() {
    entryInfoSet_.clear();
    processHistoryID_=ProcessHistoryID();
    reset_();
  }
  
  void
  BranchMapper::insert(ProductProvenance const& entryInfo) {
    //NOTE:do not read provenance here because we only need the full
    // provenance when someone tries to access it not when doing the insert
    // doing the delay saves 20% of time when doing an analysis job
    //readProvenance();
    entryInfoSet_.insert(entryInfo);
  }
    
  ProductProvenance const*
  BranchMapper::branchIDToProvenance(BranchID const& bid) const {
    readProvenance();
    ProductProvenance ei(bid);
    eiSet::const_iterator it = entryInfoSet_.find(ei);
    if(it == entryInfoSet_.end()) {
      if(nextMapper_) {
        return nextMapper_->branchIDToProvenance(bid);
      } else {
        return 0;
      }
    }
    return &*it;
  }

  BranchID
  BranchMapper::oldProductIDToBranchID_(ProductID const& ) const {
    throw edm::Exception(errors::LogicError)
        << "Internal error:  Illegal call of oldProductIDToBranchID_.\n"
        << "Please report this error to the Framework group\n";
  }
}
