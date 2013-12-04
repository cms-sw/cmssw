#include "DataFormats/Provenance/interface/ProductProvenanceRetriever.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cassert>
#include <iostream>
#include <limits>

/*
  ProductProvenanceRetriever
*/

namespace edm {
  ProductProvenanceRetriever::ProductProvenanceRetriever(unsigned int iTransitionIndex) :
      entryInfoSet_(),
      nextRetriever_(),
      provenanceReader_(),
      transitionIndex_(iTransitionIndex),
      delayedRead_(false){
  }

  ProductProvenanceRetriever::ProductProvenanceRetriever(std::unique_ptr<ProvenanceReaderBase> reader) :
      entryInfoSet_(),
      nextRetriever_(),
      provenanceReader_(reader.release()),
      transitionIndex_(std::numeric_limits<unsigned int>::max()),
      delayedRead_(true)
  {
    assert(provenanceReader_);
  }

  ProductProvenanceRetriever::~ProductProvenanceRetriever() {}

  void
  ProductProvenanceRetriever::readProvenance() const {
    if(delayedRead_ && provenanceReader_) {
      provenanceReader_->readProvenance(*this,transitionIndex_);
      delayedRead_ = false; // only read once
    }
  }

  void ProductProvenanceRetriever::deepSwap(ProductProvenanceRetriever& iFrom)
  {
    entryInfoSet_.swap(iFrom.entryInfoSet_);
    provenanceReader_ = iFrom.provenanceReader_;
    if(provenanceReader_) {
      delayedRead_=true;
    }
    if(iFrom.nextRetriever_) {
      if(not nextRetriever_) {
        nextRetriever_.reset(new ProductProvenanceRetriever(transitionIndex_));
      }
      nextRetriever_->deepSwap(*(iFrom.nextRetriever_));
    }
  }

  void
  ProductProvenanceRetriever::reset() {
    entryInfoSet_.clear();
    delayedRead_ = true;
    if(nextRetriever_) {
      nextRetriever_->reset();
    }
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
  ProductProvenanceRetriever::mergeProvenanceRetrievers(boost::shared_ptr<ProductProvenanceRetriever> other) {
    nextRetriever_ = other;
  }

  ProductProvenance const*
  ProductProvenanceRetriever::branchIDToProvenance(BranchID const& bid) const {
    readProvenance();
    ProductProvenance ei(bid);
    eiSet::const_iterator it = entryInfoSet_.find(ei);
    if(it == entryInfoSet_.end()) {
      if(nextRetriever_) {
        return nextRetriever_->branchIDToProvenance(bid);
      } else {
        return 0;
      }
    }
    return &*it;
  }

  ProvenanceReaderBase::~ProvenanceReaderBase() {
  }
}
