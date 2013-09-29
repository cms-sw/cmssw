#include "DataFormats/Provenance/interface/ProductProvenanceRetriever.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cassert>
#include <iostream>

/*
  ProductProvenanceRetriever
*/

namespace edm {
  ProductProvenanceRetriever::ProductProvenanceRetriever() :
      entryInfoSet_(),
      nextMapper_(),
      delayedRead_(false),
      provenanceReader_() {
  }

  ProductProvenanceRetriever::ProductProvenanceRetriever(std::unique_ptr<ProvenanceReaderBase> reader) :
      entryInfoSet_(),
      nextMapper_(),
      delayedRead_(true),
      provenanceReader_(reader.release()) {
    assert(provenanceReader_);
  }

  ProductProvenanceRetriever::~ProductProvenanceRetriever() {}

  void
  ProductProvenanceRetriever::readProvenance() const {
    if(delayedRead_ && provenanceReader_) {
      provenanceReader_->readProvenance(*this);
      delayedRead_ = false; // only read once
    }
  }

  void
  ProductProvenanceRetriever::reset() {
    entryInfoSet_.clear();
    delayedRead_ = true;
  }

  void
  ProductProvenanceRetriever::insertIntoSet(ProductProvenance const& entryInfo) const {
    //NOTE:do not read provenance here because we only need the full
    // provenance when someone tries to access it not when doing the insert
    // doing the delay saves 20% of time when doing an analysis job
    //readProvenance();
    entryInfoSet_.insert(entryInfo);
  }
 
  void
  ProductProvenanceRetriever::mergeMappers(boost::shared_ptr<ProductProvenanceRetriever> other) {
    nextMapper_ = other;
  }

  ProductProvenance const*
  ProductProvenanceRetriever::branchIDToProvenance(BranchID const& bid) const {
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
