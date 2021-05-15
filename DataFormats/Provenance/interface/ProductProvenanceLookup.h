#ifndef DataFormats_Provenance_ProductProvenanceLookup_h
#define DataFormats_Provenance_ProductProvenanceLookup_h

/*----------------------------------------------------------------------
  
ProductProvenanceLookup: Gives access to the per event/lumi/run per product provenance.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <vector>
#include <memory>
#include <set>
#include <atomic>

/*
 ProductProvenanceLookup
*/

namespace edm {
  class ProductRegistry;

  class ProductProvenanceLookup {
  public:
    ProductProvenanceLookup();
    explicit ProductProvenanceLookup(edm::ProductRegistry const&);
    virtual ~ProductProvenanceLookup();

    ProductProvenanceLookup& operator=(ProductProvenanceLookup const&) = delete;

    ProductProvenance const* branchIDToProvenance(BranchID const& bid) const;
    void insertIntoSet(ProductProvenance provenanceProduct) const;
    ProductProvenance const* branchIDToProvenanceForProducedOnly(BranchID const& bid) const;

    void update(edm::ProductRegistry const&);

    class ProducedProvenanceInfo {
    public:
      ProducedProvenanceInfo(BranchID iBid) : provenance_{iBid}, isParentageSet_{false} {}
      ProducedProvenanceInfo(ProducedProvenanceInfo&& iOther)
          : provenance_{std::move(iOther.provenance_)},
            isParentageSet_{iOther.isParentageSet_.load(std::memory_order_acquire)} {}
      ProducedProvenanceInfo(ProducedProvenanceInfo const& iOther) : provenance_{iOther.provenance_.branchID()} {
        bool isSet = iOther.isParentageSet_.load(std::memory_order_acquire);
        if (isSet) {
          provenance_.set(iOther.provenance_.parentageID());
        }
        isParentageSet_.store(isSet, std::memory_order_release);
      }

      ProducedProvenanceInfo& operator=(ProducedProvenanceInfo&& iOther) {
        provenance_ = std::move(iOther.provenance_);
        isParentageSet_.store(iOther.isParentageSet_.load(std::memory_order_acquire), std::memory_order_release);
        return *this;
      }
      ProducedProvenanceInfo& operator=(ProducedProvenanceInfo const& iOther) {
        bool isSet = iOther.isParentageSet_.load(std::memory_order_acquire);
        if (isSet) {
          provenance_ = iOther.provenance_;
        } else {
          provenance_ = ProductProvenance(iOther.provenance_.branchID());
        }
        isParentageSet_.store(isSet, std::memory_order_release);
        return *this;
      }

      ProductProvenance const* productProvenance() const noexcept {
        if (LIKELY(isParentageSet())) {
          return &provenance_;
        }
        return nullptr;
      }
      BranchID branchID() const noexcept { return provenance_.branchID(); }

      bool isParentageSet() const noexcept { return isParentageSet_.load(std::memory_order_acquire); }

      void threadsafe_set(ParentageID id) const {
        provenance_.set(std::move(id));
        isParentageSet_.store(true, std::memory_order_release);
      }

      void resetParentage() { isParentageSet_.store(false, std::memory_order_release); }

    private:
      CMS_THREAD_GUARD(isParentageSet_) mutable ProductProvenance provenance_;
      mutable std::atomic<bool> isParentageSet_;
    };

  protected:
    virtual std::unique_ptr<const std::set<ProductProvenance>> readProvenance() const = 0;
    virtual const ProductProvenanceLookup* nextRetriever() const = 0;

    std::vector<ProducedProvenanceInfo> entryInfoSet_;
    mutable std::atomic<const std::set<ProductProvenance>*> readEntryInfoSet_;
    edm::propagate_const<ProductProvenanceLookup const*> parentProcessRetriever_;

  private:
    void setupEntryInfoSet(edm::ProductRegistry const&);
  };
}  // namespace edm
#endif
