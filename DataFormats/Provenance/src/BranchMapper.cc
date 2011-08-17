#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cassert>
#include <iostream>

/*
  BranchMapper

*/

namespace edm {
  BranchMapper::BranchMapper() :
      entryInfoSet_(),
      nextMapper_(),
      delayedRead_(false),
      provenanceReader_() {
  }

  BranchMapper::BranchMapper(boost::shared_ptr<ProvenanceReaderBase> reader) :
      entryInfoSet_(),
      nextMapper_(),
      delayedRead_(true),
      provenanceReader_(reader) {
    assert(reader);
  }

  BranchMapper::~BranchMapper() {}

  void
  BranchMapper::readProvenance() const {
    if(delayedRead_ && provenanceReader_) {
      provenanceReader_->readProvenance(*this);
      delayedRead_ = false; // only read once
    }
  }

  void
  BranchMapper::reset() {
    entryInfoSet_.clear();
    delayedRead_ = true;
  }
  
  void
  BranchMapper::insertIntoSet(ProductProvenance const& entryInfo) const {
    //NOTE:do not read provenance here because we only need the full
    // provenance when someone tries to access it not when doing the insert
    // doing the delay saves 20% of time when doing an analysis job
    //readProvenance();
    entryInfoSet_.insert(entryInfo);
  }
    
  void
  BranchMapper::mergeMappers(boost::shared_ptr<BranchMapper> other) {
    nextMapper_ = other;
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

  ProvenanceReaderBase::~ProvenanceReaderBase() {
  }
}
