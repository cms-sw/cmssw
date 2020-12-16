#include "DataFormats/Provenance/interface/ProductProvenanceRetriever.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cassert>
#include <iostream>
#include <limits>

/*
  ProductProvenanceRetriever
*/

namespace edm {
  ProductProvenanceRetriever::ProductProvenanceRetriever(unsigned int iTransitionIndex)
      : entryInfoSet_(),
        readEntryInfoSet_(),
        nextRetriever_(),
        parentProcessRetriever_(nullptr),
        provenanceReader_(),
        transitionIndex_(iTransitionIndex) {}

  ProductProvenanceRetriever::ProductProvenanceRetriever(unsigned int iTransitionIndex,
                                                         edm::ProductRegistry const& iReg)
      : entryInfoSet_(),
        readEntryInfoSet_(),
        nextRetriever_(),
        parentProcessRetriever_(nullptr),
        provenanceReader_(),
        transitionIndex_(iTransitionIndex) {
    setupEntryInfoSet(iReg);
  }

  ProductProvenanceRetriever::ProductProvenanceRetriever(std::unique_ptr<ProvenanceReaderBase> reader)
      : entryInfoSet_(),
        readEntryInfoSet_(),
        nextRetriever_(),
        parentProcessRetriever_(nullptr),
        provenanceReader_(reader.release()),
        transitionIndex_(std::numeric_limits<unsigned int>::max()) {
    assert(provenanceReader_);
  }

  ProductProvenanceRetriever::~ProductProvenanceRetriever() { delete readEntryInfoSet_.load(); }

  void ProductProvenanceRetriever::readProvenance() const {
    if (nullptr == readEntryInfoSet_.load() && provenanceReader_) {
      auto temp =
          std::make_unique<std::set<ProductProvenance> const>(provenanceReader_->readProvenance(transitionIndex_));
      std::set<ProductProvenance> const* expected = nullptr;
      if (readEntryInfoSet_.compare_exchange_strong(expected, temp.get())) {
        temp.release();
      }
    }
  }

  void ProductProvenanceRetriever::setupEntryInfoSet(edm::ProductRegistry const& iReg) {
    std::set<BranchID> ids;
    for (auto const& p : iReg.productList()) {
      if (p.second.branchType() == edm::InEvent) {
        if (p.second.produced() or p.second.isProvenanceSetOnRead()) {
          ids.insert(p.second.branchID());
        }
      }
    }
    entryInfoSet_.reserve(ids.size());
    for (auto const& b : ids) {
      entryInfoSet_.emplace_back(b);
    }
  }

  void ProductProvenanceRetriever::update(edm::ProductRegistry const& iReg) {
    entryInfoSet_.clear();
    setupEntryInfoSet(iReg);
  }

  void ProductProvenanceRetriever::readProvenanceAsync(WaitingTaskHolder task,
                                                       ModuleCallingContext const* moduleCallingContext) const {
    if (provenanceReader_ and nullptr == readEntryInfoSet_.load()) {
      provenanceReader_->readProvenanceAsync(task, moduleCallingContext, transitionIndex_, readEntryInfoSet_);
    }
    if (nextRetriever_) {
      nextRetriever_->readProvenanceAsync(task, moduleCallingContext);
    }
  }

  void ProductProvenanceRetriever::deepCopy(ProductProvenanceRetriever const& iFrom) {
    if (iFrom.readEntryInfoSet_) {
      if (readEntryInfoSet_) {
        delete readEntryInfoSet_.exchange(nullptr);
      }
      readEntryInfoSet_ = new std::set<ProductProvenance>(*iFrom.readEntryInfoSet_);
    } else {
      if (readEntryInfoSet_) {
        delete readEntryInfoSet_.load();
        readEntryInfoSet_ = nullptr;
      }
    }
    assert(iFrom.entryInfoSet_.empty());
    provenanceReader_ = iFrom.provenanceReader_;

    if (iFrom.nextRetriever_) {
      if (not nextRetriever_) {
        nextRetriever_ = std::make_shared<ProductProvenanceRetriever>(transitionIndex_);
      }
      nextRetriever_->deepCopy(*(iFrom.nextRetriever_));
    }
  }

  void ProductProvenanceRetriever::reset() {
    delete readEntryInfoSet_.load();
    readEntryInfoSet_ = nullptr;
    for (auto& e : entryInfoSet_) {
      e.resetParentage();
    }
    if (nextRetriever_) {
      nextRetriever_->reset();
    }
    parentProcessRetriever_ = nullptr;
  }

  void ProductProvenanceRetriever::insertIntoSet(ProductProvenance entryInfo) const {
    //NOTE:do not read provenance here because we only need the full
    // provenance when someone tries to access it not when doing the insert
    // doing the delay saves 20% of time when doing an analysis job
    //readProvenance();
    auto itFound =
        std::lower_bound(entryInfoSet_.begin(),
                         entryInfoSet_.end(),
                         entryInfo.branchID(),
                         [](auto const& iEntry, edm::BranchID const& iValue) { return iEntry.branchID() < iValue; });
    if UNLIKELY (itFound == entryInfoSet_.end() or itFound->branchID() != entryInfo.branchID()) {
      throw edm::Exception(edm::errors::LogicError) << "ProductProvenanceRetriever::insertIntoSet passed a BranchID "
                                                    << entryInfo.branchID().id() << " that has not been pre-registered";
    }
    itFound->threadsafe_set(entryInfo.moveParentageID());
  }

  void ProductProvenanceRetriever::mergeProvenanceRetrievers(std::shared_ptr<ProductProvenanceRetriever> other) {
    nextRetriever_ = other;
  }

  void ProductProvenanceRetriever::mergeParentProcessRetriever(ProductProvenanceRetriever const& provRetriever) {
    parentProcessRetriever_ = &provRetriever;
  }

  ProductProvenance const* ProductProvenanceRetriever::branchIDToProvenance(BranchID const& bid) const {
    auto itFound = std::lower_bound(
        entryInfoSet_.begin(), entryInfoSet_.end(), bid, [](auto const& iEntry, edm::BranchID const& iValue) {
          return iEntry.branchID() < iValue;
        });
    if (itFound != entryInfoSet_.end() and itFound->branchID() == bid) {
      if (auto p = itFound->productProvenance()) {
        return p;
      }
    }
    if (parentProcessRetriever_) {
      return parentProcessRetriever_->branchIDToProvenance(bid);
    }
    //check in source
    readProvenance();
    auto ptr = readEntryInfoSet_.load();
    if (ptr) {
      ProductProvenance ei(bid);
      auto itRead = ptr->find(ei);
      if (itRead != ptr->end()) {
        return &*itRead;
      }
    }
    if (nextRetriever_) {
      return nextRetriever_->branchIDToProvenance(bid);
    }
    return nullptr;
  }

  ProductProvenance const* ProductProvenanceRetriever::branchIDToProvenanceForProducedOnly(BranchID const& bid) const {
    auto itFound = std::lower_bound(
        entryInfoSet_.begin(), entryInfoSet_.end(), bid, [](auto const& iEntry, edm::BranchID const& iValue) {
          return iEntry.branchID() < iValue;
        });
    if (itFound != entryInfoSet_.end() and itFound->branchID() == bid) {
      if (auto p = itFound->productProvenance()) {
        return p;
      }
    }
    if (parentProcessRetriever_) {
      return parentProcessRetriever_->branchIDToProvenanceForProducedOnly(bid);
    }
    if (nextRetriever_) {
      return nextRetriever_->branchIDToProvenanceForProducedOnly(bid);
    }
    return nullptr;
  }

  ProvenanceReaderBase::~ProvenanceReaderBase() {}
}  // namespace edm
