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
      readEntryInfoSet_(),
      nextRetriever_(),
      provenanceReader_(),
      transitionIndex_(iTransitionIndex){
  }

  ProductProvenanceRetriever::ProductProvenanceRetriever(std::unique_ptr<ProvenanceReaderBase> reader) :
      entryInfoSet_(),
      readEntryInfoSet_(),
      nextRetriever_(),
      provenanceReader_(reader.release()),
      transitionIndex_(std::numeric_limits<unsigned int>::max())
  {
    assert(provenanceReader_);
  }

  ProductProvenanceRetriever::~ProductProvenanceRetriever() {
    delete readEntryInfoSet_.load();
  }

  void
  ProductProvenanceRetriever::readProvenance() const {
    if(nullptr == readEntryInfoSet_.load() && provenanceReader_) {
      auto temp = std::make_unique<std::set<ProductProvenance> const>(provenanceReader_->readProvenance(transitionIndex_));
      std::set<ProductProvenance> const* expected = nullptr;
      if(readEntryInfoSet_.compare_exchange_strong(expected, temp.get())) {
        temp.release();
      }
    }
  }

  void ProductProvenanceRetriever::deepCopy(ProductProvenanceRetriever const& iFrom)
  {
    if(iFrom.readEntryInfoSet_) {
      if (readEntryInfoSet_) {
        delete readEntryInfoSet_.exchange(nullptr);
      }
      readEntryInfoSet_ = new std::set<ProductProvenance>(*iFrom.readEntryInfoSet_);
    } else {
      if(readEntryInfoSet_) {
        delete readEntryInfoSet_.load();
        readEntryInfoSet_ = nullptr;
      }
    }
    entryInfoSet_ = iFrom.entryInfoSet_;
    provenanceReader_ = iFrom.provenanceReader_;
    
    if(iFrom.nextRetriever_) {
      if(not nextRetriever_) {
        get_underlying(nextRetriever_) = std::make_shared<ProductProvenanceRetriever>(transitionIndex_);
      }
      nextRetriever_->deepCopy(*(iFrom.nextRetriever_));
    }
  }

  void
  ProductProvenanceRetriever::reset() {
    delete readEntryInfoSet_.load();
    readEntryInfoSet_ = nullptr;
    entryInfoSet_.clear();
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
  ProductProvenanceRetriever::mergeProvenanceRetrievers(std::shared_ptr<ProductProvenanceRetriever> other) {
    nextRetriever_ = other;
  }

  ProductProvenance const*
  ProductProvenanceRetriever::branchIDToProvenance(BranchID const& bid) const {
    ProductProvenance ei(bid);
    auto it = entryInfoSet_.find(ei);
    if(it == entryInfoSet_.end()) {
      //check in source
      readProvenance();
      auto ptr =readEntryInfoSet_.load();
      if(ptr) {
        auto it = ptr->find(ei);
        if(it!= ptr->end()) {
          return &*it;
        }
      }
      if(nextRetriever_) {
        return nextRetriever_->branchIDToProvenance(bid);
      }
      return nullptr;
    }
    return &*it;
  }

  ProvenanceReaderBase::~ProvenanceReaderBase() {
  }
}
