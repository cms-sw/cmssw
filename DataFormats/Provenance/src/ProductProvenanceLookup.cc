#include "DataFormats/Provenance/interface/ProductProvenanceLookup.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>

/*
 ProductProvenanceLookup
*/

namespace edm {
  ProductProvenanceLookup::ProductProvenanceLookup()
      : entryInfoSet_(), readEntryInfoSet_(), parentProcessRetriever_(nullptr) {}

  ProductProvenanceLookup::ProductProvenanceLookup(edm::ProductRegistry const& iReg)
      : entryInfoSet_(), readEntryInfoSet_(), parentProcessRetriever_(nullptr) {
    setupEntryInfoSet(iReg);
  }

  ProductProvenanceLookup::~ProductProvenanceLookup() { delete readEntryInfoSet_.load(); }

  void ProductProvenanceLookup::setupEntryInfoSet(edm::ProductRegistry const& iReg) {
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

  void ProductProvenanceLookup::update(edm::ProductRegistry const& iReg) {
    entryInfoSet_.clear();
    setupEntryInfoSet(iReg);
  }

  void ProductProvenanceLookup::insertIntoSet(ProductProvenance entryInfo) const {
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
      throw edm::Exception(edm::errors::LogicError) << "ProductProvenanceLookup::insertIntoSet passed a BranchID "
                                                    << entryInfo.branchID().id() << " that has not been pre-registered";
    }
    itFound->threadsafe_set(entryInfo.moveParentageID());
  }

  ProductProvenance const* ProductProvenanceLookup::branchIDToProvenance(BranchID const& bid) const {
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
    if (nullptr == readEntryInfoSet_.load()) {
      auto readProv = readProvenance();
      std::set<ProductProvenance> const* expected = nullptr;
      if (readEntryInfoSet_.compare_exchange_strong(expected, readProv.get())) {
        readProv.release();
      }
    }
    auto ptr = readEntryInfoSet_.load();
    if (ptr) {
      ProductProvenance ei(bid);
      auto itRead = ptr->find(ei);
      if (itRead != ptr->end()) {
        return &*itRead;
      }
    }
    auto nr = nextRetriever();
    if (nr) {
      return nr->branchIDToProvenance(bid);
    }
    return nullptr;
  }

  ProductProvenance const* ProductProvenanceLookup::branchIDToProvenanceForProducedOnly(BranchID const& bid) const {
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
    auto nr = nextRetriever();
    if (nr) {
      return nr->branchIDToProvenanceForProducedOnly(bid);
    }
    return nullptr;
  }

}  // namespace edm
