#ifndef DataFormats_Provenance_BranchMapper_h
#define DataFormats_Provenance_BranchMapper_h

/*----------------------------------------------------------------------
  
ProductProvenanceRetriever: Manages the per event/lumi/run per product provenance.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <vector>
#include <memory>
#include <set>
#include <atomic>
#include <string_view>

/*
  ProductProvenanceRetriever
*/

namespace edm {
  class ProvenanceReaderBase;
  class WaitingTask;
  class ModuleCallingContext;
  class ProductRegistry;

  struct ProductProvenanceHasher {
    size_t operator()(ProductProvenance const& tid) const { return tid.branchID().id(); }
  };

  struct ProductProvenanceEqual {
    //The default operator== for ProductProvenance does not work for this case
    // Since (a<b)==false  and (a>b)==false does not mean a==b for the operators
    // defined for ProductProvenance
    bool operator()(ProductProvenance const& iLHS, ProductProvenance const iRHS) const {
      return iLHS.branchID().id() == iRHS.branchID().id();
    }
  };

  class ProvenanceReaderBase {
  public:
    ProvenanceReaderBase() {}
    virtual ~ProvenanceReaderBase();
    virtual std::set<ProductProvenance> readProvenance(unsigned int transitionIndex) const = 0;
    virtual void readProvenanceAsync(WaitingTask* task,
                                     ModuleCallingContext const* moduleCallingContext,
                                     unsigned int transitionIndex,
                                     std::atomic<const std::set<ProductProvenance>*>& writeTo) const = 0;
  };

  class ProductProvenanceRetriever {
  public:
    explicit ProductProvenanceRetriever(unsigned int iTransitionIndex);
    ProductProvenanceRetriever(unsigned int iTransitionIndex, edm::ProductRegistry const&);
    explicit ProductProvenanceRetriever(std::unique_ptr<ProvenanceReaderBase> reader);

    ProductProvenanceRetriever& operator=(ProductProvenanceRetriever const&) = delete;

    ~ProductProvenanceRetriever();

    ProductProvenance const* branchIDToProvenance(BranchID const& bid) const;

    ProductProvenance const* branchIDToProvenanceForProducedOnly(BranchID const& bid) const;

    void insertIntoSet(ProductProvenance provenanceProduct) const;

    void mergeProvenanceRetrievers(std::shared_ptr<ProductProvenanceRetriever> other);

    void mergeParentProcessRetriever(ProductProvenanceRetriever const& provRetriever);

    void deepCopy(ProductProvenanceRetriever const&);

    void reset();

    void readProvenanceAsync(WaitingTask* task, ModuleCallingContext const* moduleCallingContext) const;

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

  private:
    void readProvenance() const;
    void setTransitionIndex(unsigned int transitionIndex) { transitionIndex_ = transitionIndex; }
    void setupEntryInfoSet(edm::ProductRegistry const&);

    std::vector<ProducedProvenanceInfo> entryInfoSet_;
    mutable std::atomic<const std::set<ProductProvenance>*> readEntryInfoSet_;
    edm::propagate_const<std::shared_ptr<ProductProvenanceRetriever>> nextRetriever_;
    edm::propagate_const<ProductProvenanceRetriever const*> parentProcessRetriever_;
    std::shared_ptr<const ProvenanceReaderBase> provenanceReader_;
    unsigned int transitionIndex_;
  };

}  // namespace edm
#endif
