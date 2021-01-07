#include "FWCore/Framework/interface/EventPrincipal.h"

#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/FunctorHandleExceptionFactory.h"
#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/getThinned_implementation.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "DataFormats/Provenance/interface/ProductIDToBranchID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "FWCore/Framework/interface/ProductResolverBase.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/src/ProductDeletedException.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>

namespace edm {
  EventPrincipal::EventPrincipal(std::shared_ptr<ProductRegistry const> reg,
                                 std::shared_ptr<BranchIDListHelper const> branchIDListHelper,
                                 std::shared_ptr<ThinnedAssociationsHelper const> thinnedAssociationsHelper,
                                 ProcessConfiguration const& pc,
                                 HistoryAppender* historyAppender,
                                 unsigned int streamIndex,
                                 bool isForPrimaryProcess)
      : Base(reg, reg->productLookup(InEvent), pc, InEvent, historyAppender, isForPrimaryProcess),
        aux_(),
        luminosityBlockPrincipal_(nullptr),
        provRetrieverPtr_(new ProductProvenanceRetriever(streamIndex, *reg)),
        eventSelectionIDs_(),
        branchIDListHelper_(branchIDListHelper),
        thinnedAssociationsHelper_(thinnedAssociationsHelper),
        branchListIndexes_(),
        branchListIndexToProcessIndex_(),
        streamID_(streamIndex) {
    assert(thinnedAssociationsHelper_);

    for (auto& prod : *this) {
      if (prod->singleProduct()) {
        prod->setProductProvenanceRetriever(productProvenanceRetrieverPtr());
      }
    }
  }

  void EventPrincipal::clearEventPrincipal() {
    clearPrincipal();
    aux_ = EventAuxiliary();
    //do not clear luminosityBlockPrincipal_ since
    // it is only connected at beginLumi transition
    provRetrieverPtr_->reset();
  }

  void EventPrincipal::fillEventPrincipal(EventAuxiliary const& aux,
                                          ProcessHistory const* processHistory,
                                          EventSelectionIDVector eventSelectionIDs,
                                          BranchListIndexes branchListIndexes,
                                          ProductProvenanceRetriever const& provRetriever,
                                          DelayedReader* reader,
                                          bool deepCopyRetriever) {
    eventSelectionIDs_ = std::move(eventSelectionIDs);
    if (deepCopyRetriever) {
      provRetrieverPtr_->deepCopy(provRetriever);
    } else {
      provRetrieverPtr_->mergeParentProcessRetriever(provRetriever);
    }
    if (wasBranchListIndexesChangedFromInput(branchListIndexes)) {
      if (branchIDListHelper_->hasProducedProducts()) {
        // Add index into BranchIDListRegistry for products produced this process
        branchListIndexes.push_back(branchIDListHelper_->producedBranchListIndex());
      }
      updateBranchListIndexes(std::move(branchListIndexes));
    }
    commonFillEventPrincipal(aux, processHistory, reader);
  }

  void EventPrincipal::fillEventPrincipal(EventAuxiliary const& aux,
                                          ProcessHistory const* processHistory,
                                          EventSelectionIDVector eventSelectionIDs,
                                          BranchListIndexes branchListIndexes) {
    eventSelectionIDs_ = std::move(eventSelectionIDs);

    if (wasBranchListIndexesChangedFromInput(branchListIndexes)) {
      if (branchIDListHelper_->hasProducedProducts()) {
        // Add index into BranchIDListRegistry for products produced this process
        branchListIndexes.push_back(branchIDListHelper_->producedBranchListIndex());
      }
      updateBranchListIndexes(std::move(branchListIndexes));
    }
    commonFillEventPrincipal(aux, processHistory, nullptr);
  }

  void EventPrincipal::fillEventPrincipal(EventAuxiliary const& aux,
                                          ProcessHistory const* processHistory,
                                          DelayedReader* reader) {
    if (branchListIndexes_.empty() and branchIDListHelper_->hasProducedProducts()) {
      // Add index into BranchIDListRegistry for products produced this process
      //  if it hasn't already been filled in by the other fillEventPrincipal or by an earlier call to this function
      BranchListIndexes indexes;
      indexes.push_back(branchIDListHelper_->producedBranchListIndex());
      updateBranchListIndexes(std::move(indexes));
    }
    commonFillEventPrincipal(aux, processHistory, reader);
  }

  void EventPrincipal::commonFillEventPrincipal(EventAuxiliary const& aux,
                                                ProcessHistory const* processHistory,
                                                DelayedReader* reader) {
    if (aux.event() == invalidEventNumber) {
      throw Exception(errors::LogicError) << "EventPrincipal::fillEventPrincipal, Invalid event number provided in "
                                             "EventAuxiliary, It is illegal for the event number to be 0\n";
    }

    fillPrincipal(aux.processHistoryID(), processHistory, reader);
    aux_ = aux;
    aux_.setProcessHistoryID(processHistoryID());
  }

  bool EventPrincipal::wasBranchListIndexesChangedFromInput(BranchListIndexes const& fromInput) const {
    //fromInput does not contain entries for what is being produced in this job.
    auto end = branchListIndexes_.end();
    if (end != branchListIndexes_.begin() and branchIDListHelper_->hasProducedProducts()) {
      --end;
    }

    return not std::equal(fromInput.begin(), fromInput.end(), branchListIndexes_.begin(), end);
  }

  void EventPrincipal::updateBranchListIndexes(BranchListIndexes&& branchListIndexes) {
    branchListIndexes_ = std::move(branchListIndexes);
    branchListIndexToProcessIndex_.clear();
    // Fill in helper map for Branch to ProductID mapping
    if (not branchListIndexes_.empty()) {
      ProcessIndex pix = 0;
      branchListIndexToProcessIndex_.resize(1 + *std::max_element(branchListIndexes_.begin(), branchListIndexes_.end()),
                                            std::numeric_limits<BranchListIndex>::max());
      for (auto const& blindex : branchListIndexes_) {
        branchListIndexToProcessIndex_[blindex] = pix;
        ++pix;
      }
    }

    // Fill in the product ID's in the product holders.
    for (auto& prod : *this) {
      if (prod->singleProduct()) {
        // If an alias is in the same process as the original then isAlias will be true.
        //  Under that condition, we want the ProductID to be the same as the original.
        //  If not, then we've internally changed the original BranchID to the alias BranchID
        //  in the ProductID lookup so we need the alias BranchID.

        auto const& bd = prod->branchDescription();
        prod->setProductID(branchIDToProductID(bd.isAlias() ? bd.originalBranchID() : bd.branchID()));
      }
    }
  }

  void EventPrincipal::setLuminosityBlockPrincipal(LuminosityBlockPrincipal* lbp) { luminosityBlockPrincipal_ = lbp; }

  void EventPrincipal::setRunAndLumiNumber(RunNumber_t run, LuminosityBlockNumber_t lumi) {
    assert(run == luminosityBlockPrincipal_->run());
    assert(lumi == luminosityBlockPrincipal_->luminosityBlock());
    EventNumber_t event = aux_.id().event();
    aux_.id() = EventID(run, lumi, event);
  }

  RunPrincipal const& EventPrincipal::runPrincipal() const { return luminosityBlockPrincipal().runPrincipal(); }

  void EventPrincipal::put(BranchDescription const& bd,
                           std::unique_ptr<WrapperBase> edp,
                           ProductProvenance const& productProvenance) const {
    // assert commented out for DaqSource.  When DaqSource no longer uses put(), the assert can be restored.
    //assert(produced());
    if (edp.get() == nullptr) {
      throw Exception(errors::InsertFailure, "Null Pointer") << "put: Cannot put because ptr to product is null."
                                                             << "\n";
    }
    productProvenanceRetrieverPtr()->insertIntoSet(productProvenance);
    auto phb = getExistingProduct(bd.branchID());
    assert(phb);
    // ProductResolver assumes ownership
    phb->putProduct(std::move(edp));
  }

  void EventPrincipal::put(ProductResolverIndex index, std::unique_ptr<WrapperBase> edp, ParentageID parentage) const {
    if (edp.get() == nullptr) {
      throw Exception(errors::InsertFailure, "Null Pointer") << "put: Cannot put because ptr to product is null."
                                                             << "\n";
    }
    auto phb = getProductResolverByIndex(index);

    productProvenanceRetrieverPtr()->insertIntoSet(
        ProductProvenance(phb->branchDescription().branchID(), std::move(parentage)));

    assert(phb);
    // ProductResolver assumes ownership
    phb->putProduct(std::move(edp));
  }

  void EventPrincipal::putOnRead(BranchDescription const& bd,
                                 std::unique_ptr<WrapperBase> edp,
                                 std::optional<ProductProvenance> productProvenance) const {
    assert(!bd.produced());
    if (productProvenance) {
      productProvenanceRetrieverPtr()->insertIntoSet(std::move(*productProvenance));
    }
    auto phb = getExistingProduct(bd.branchID());
    assert(phb);
    // ProductResolver assumes ownership
    phb->putProduct(std::move(edp));
  }

  BranchID EventPrincipal::pidToBid(ProductID const& pid) const {
    if (!pid.isValid()) {
      throw Exception(errors::ProductNotFound, "InvalidID") << "get by product ID: invalid ProductID supplied\n";
    }
    return productIDToBranchID(pid, branchIDListHelper_->branchIDLists(), branchListIndexes_);
  }

  ProductID EventPrincipal::branchIDToProductID(BranchID const& bid) const {
    if (!bid.isValid()) {
      throw Exception(errors::NotFound, "InvalidID") << "branchIDToProductID: invalid BranchID supplied\n";
    }
    typedef BranchIDListHelper::BranchIDToIndexMap BIDToIndexMap;
    typedef BIDToIndexMap::const_iterator Iter;
    typedef std::pair<Iter, Iter> IndexRange;

    IndexRange range = branchIDListHelper_->branchIDToIndexMap().equal_range(bid);
    for (Iter it = range.first; it != range.second; ++it) {
      BranchListIndex blix = it->second.first;
      if (blix < branchListIndexToProcessIndex_.size()) {
        auto v = branchListIndexToProcessIndex_[blix];
        if (v != std::numeric_limits<BranchListIndex>::max()) {
          ProductIndex productIndex = it->second.second;
          ProcessIndex processIndex = v;
          return ProductID(processIndex + 1, productIndex + 1);
        }
      }
    }
    // cannot throw, because some products may legitimately not have product ID's (e.g. pile-up).
    return ProductID();
  }

  unsigned int EventPrincipal::transitionIndex_() const { return streamID_.value(); }

  void EventPrincipal::changedIndexes_() { provRetrieverPtr_->update(productRegistry()); }

  static void throwProductDeletedException(ProductID const& pid,
                                           edm::EventPrincipal::ConstProductResolverPtr const phb) {
    ProductDeletedException exception;
    exception << "get by product ID: The product with given id: " << pid << "\ntype: " << phb->productType()
              << "\nproduct instance name: " << phb->productInstanceName() << "\nprocess name: " << phb->processName()
              << "\nwas already deleted. This is a configuration error. Please change the configuration of the module "
                 "which caused this exception to state it reads this data.";
    throw exception;
  }

  BasicHandle EventPrincipal::getByProductID(ProductID const& pid) const {
    BranchID bid = pidToBid(pid);
    ConstProductResolverPtr const phb = getProductResolver(bid);
    if (phb == nullptr) {
      return BasicHandle(makeHandleExceptionFactory([pid]() -> std::shared_ptr<cms::Exception> {
        std::shared_ptr<cms::Exception> whyFailed(std::make_shared<Exception>(errors::ProductNotFound, "InvalidID"));
        *whyFailed << "get by product ID: no product with given id: " << pid << "\n";
        return whyFailed;
      }));
    }

    // Was this already deleted?
    if (phb->productWasDeleted()) {
      throwProductDeletedException(pid, phb);
    }
    // Check for case where we tried on demand production and
    // it failed to produce the object
    if (phb->unscheduledWasNotRun()) {
      return BasicHandle(makeHandleExceptionFactory([pid]() -> std::shared_ptr<cms::Exception> {
        std::shared_ptr<cms::Exception> whyFailed(std::make_shared<Exception>(errors::ProductNotFound, "InvalidID"));
        *whyFailed << "get by ProductID: could not get product with id: " << pid << "\n"
                   << "Unscheduled execution not allowed to get via ProductID.\n";
        return whyFailed;
      }));
    }
    auto resolution = phb->resolveProduct(*this, false, nullptr, nullptr);

    auto data = resolution.data();
    if (data) {
      return BasicHandle(data->wrapper(), &(data->provenance()));
    }
    return BasicHandle(nullptr, nullptr);
  }

  WrapperBase const* EventPrincipal::getIt(ProductID const& pid) const { return getByProductID(pid).wrapper(); }

  std::optional<std::tuple<WrapperBase const*, unsigned int>> EventPrincipal::getThinnedProduct(
      ProductID const& pid, unsigned int key) const {
    return detail::getThinnedProduct(
        pid,
        key,
        *thinnedAssociationsHelper_,
        [this](ProductID const& p) { return pidToBid(p); },
        [this](BranchID const& b) { return getThinnedAssociation(b); },
        [this](ProductID const& p) { return getIt(p); });
  }

  void EventPrincipal::getThinnedProducts(ProductID const& pid,
                                          std::vector<WrapperBase const*>& foundContainers,
                                          std::vector<unsigned int>& keys) const {
    detail::getThinnedProducts(
        pid,
        *thinnedAssociationsHelper_,
        [this](ProductID const& p) { return pidToBid(p); },
        [this](BranchID const& b) { return getThinnedAssociation(b); },
        [this](ProductID const& p) { return getIt(p); },
        foundContainers,
        keys);
  }

  OptionalThinnedKey EventPrincipal::getThinnedKeyFrom(ProductID const& parentID,
                                                       unsigned int key,
                                                       ProductID const& thinnedID) const {
    BranchID parent = pidToBid(parentID);
    BranchID thinned = pidToBid(thinnedID);

    try {
      auto ret = detail::getThinnedKeyFrom_implementation(
          parentID, parent, key, thinnedID, thinned, *thinnedAssociationsHelper_, [this](BranchID const& branchID) {
            return getThinnedAssociation(branchID);
          });
      if (auto factory = std::get_if<detail::GetThinnedKeyFromExceptionFactory>(&ret)) {
        return [func = *factory]() {
          auto ex = func();
          ex.addContext("Calling EventPrincipal::getThinnedKeyFrom()");
          return ex;
        };
      } else {
        return ret;
      }
    } catch (Exception& ex) {
      ex.addContext("Calling EventPrincipal::getThinnedKeyFrom()");
      throw ex;
    }
  }

  Provenance const& EventPrincipal::getProvenance(ProductID const& pid) const {
    BranchID bid = pidToBid(pid);
    return getProvenance(bid);
  }

  StableProvenance const& EventPrincipal::getStableProvenance(ProductID const& pid) const {
    BranchID bid = pidToBid(pid);
    return getStableProvenance(bid);
  }

  EventSelectionIDVector const& EventPrincipal::eventSelectionIDs() const { return eventSelectionIDs_; }

  BranchListIndexes const& EventPrincipal::branchListIndexes() const { return branchListIndexes_; }

  edm::ThinnedAssociation const* EventPrincipal::getThinnedAssociation(edm::BranchID const& branchID) const {
    ConstProductResolverPtr const phb = getProductResolver(branchID);

    if (phb == nullptr) {
      throw Exception(errors::LogicError)
          << "EventPrincipal::getThinnedAssociation, ThinnedAssociation ProductResolver cannot be found\n"
          << "This should never happen. Contact a Framework developer";
    }
    ProductData const* productData = (phb->resolveProduct(*this, false, nullptr, nullptr)).data();
    if (productData == nullptr) {
      return nullptr;
    }
    WrapperBase const* product = productData->wrapper();
    if (!(typeid(edm::ThinnedAssociation) == product->dynamicTypeInfo())) {
      throw Exception(errors::LogicError)
          << "EventPrincipal::getThinnedProduct, product has wrong type, not a ThinnedAssociation.\n";
    }
    Wrapper<ThinnedAssociation> const* wrapper = static_cast<Wrapper<ThinnedAssociation> const*>(product);
    return wrapper->product();
  }

}  // namespace edm
