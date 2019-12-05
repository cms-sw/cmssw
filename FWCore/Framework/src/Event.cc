#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/StableProvenance.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace {
  const edm::ParentageID s_emptyParentage;
}
namespace edm {

  std::string const Event::emptyString_;

  Event::Event(EventPrincipal const& ep, ModuleDescription const& md, ModuleCallingContext const* moduleCallingContext)
      : provRecorder_(ep, md, true /*always at end*/),
        aux_(ep.aux()),
        luminosityBlock_(),
        gotBranchIDs_(),
        gotViews_(),
        streamID_(ep.streamID()),
        moduleCallingContext_(moduleCallingContext) {}

  Event::~Event() {}

  Event::CacheIdentifier_t Event::cacheIdentifier() const { return eventPrincipal().cacheIdentifier(); }

  void Event::setConsumer(EDConsumerBase const* iConsumer) {
    provRecorder_.setConsumer(iConsumer);
    gotBranchIDs_.reserve(provRecorder_.numberOfProductsConsumed());
    if (luminosityBlock_) {
      luminosityBlock_->setConsumer(iConsumer);
    }
  }

  void Event::setSharedResourcesAcquirer(SharedResourcesAcquirer* iResourceAcquirer) {
    provRecorder_.setSharedResourcesAcquirer(iResourceAcquirer);
    if (luminosityBlock_) {
      luminosityBlock_->setSharedResourcesAcquirer(iResourceAcquirer);
    }
  }

  void Event::fillLuminosityBlock() const {
    luminosityBlock_.emplace(
        eventPrincipal().luminosityBlockPrincipal(), provRecorder_.moduleDescription(), moduleCallingContext_, false);
    luminosityBlock_->setConsumer(provRecorder_.getConsumer());
    luminosityBlock_->setSharedResourcesAcquirer(provRecorder_.getSharedResourcesAcquirer());
  }

  void Event::setProducerCommon(ProducerBase const* iProd, std::vector<BranchID>* previousParentage) {
    provRecorder_.setProducer(iProd);
    //set appropriate size
    putProducts_.resize(provRecorder_.putTokenIndexToProductResolverIndex().size());
    previousBranchIDs_ = previousParentage;
  }

  void Event::setProducer(ProducerBase const* iProd,
                          std::vector<BranchID>* previousParentage,
                          std::vector<BranchID>* gotBranchIDsFromAcquire) {
    setProducerCommon(iProd, previousParentage);
    if (previousParentage) {
      //are we supposed to record parentage for at least one item?
      bool record_parents = false;
      for (auto v : provRecorder_.recordProvenanceList()) {
        if (v) {
          record_parents = true;
          break;
        }
      }
      if (not record_parents) {
        previousBranchIDs_ = nullptr;
        return;
      }
      gotBranchIDsFromPrevious_.resize(previousParentage->size(), false);
      if (gotBranchIDsFromAcquire) {
        for (auto const& branchID : *gotBranchIDsFromAcquire) {
          addToGotBranchIDs(branchID);
        }
      }
    }
  }

  void Event::setProducerForAcquire(ProducerBase const* iProd,
                                    std::vector<BranchID>* previousParentage,
                                    std::vector<BranchID>& gotBranchIDsFromAcquire) {
    setProducerCommon(iProd, previousParentage);
    gotBranchIDsFromAcquire_ = &gotBranchIDsFromAcquire;
    gotBranchIDsFromAcquire_->clear();
  }

  EventPrincipal const& Event::eventPrincipal() const {
    return dynamic_cast<EventPrincipal const&>(provRecorder_.principal());
  }

  EDProductGetter const& Event::productGetter() const { return provRecorder_.principal(); }

  ProductID Event::makeProductID(BranchDescription const& desc) const {
    return eventPrincipal().branchIDToProductID(desc.originalBranchID());
  }

  Run const& Event::getRun() const { return getLuminosityBlock().getRun(); }

  EventSelectionIDVector const& Event::eventSelectionIDs() const { return eventPrincipal().eventSelectionIDs(); }

  ProcessHistoryID const& Event::processHistoryID() const { return eventPrincipal().processHistoryID(); }

  Provenance Event::getProvenance(BranchID const& bid) const {
    return provRecorder_.principal().getProvenance(bid, moduleCallingContext_);
  }

  Provenance Event::getProvenance(ProductID const& pid) const {
    return eventPrincipal().getProvenance(pid, moduleCallingContext_);
  }

  void Event::getAllProvenance(std::vector<Provenance const*>& provenances) const {
    provRecorder_.principal().getAllProvenance(provenances);
  }

  void Event::getAllStableProvenance(std::vector<StableProvenance const*>& provenances) const {
    provRecorder_.principal().getAllStableProvenance(provenances);
  }

  bool Event::getProcessParameterSet(std::string const& processName, ParameterSet& ps) const {
    ProcessConfiguration config;
    bool process_found = processHistory().getConfigurationForProcess(processName, config);
    if (process_found) {
      pset::Registry::instance()->getMapped(config.parameterSetID(), ps);
      assert(!ps.empty());
    }
    return process_found;
  }

  edm::ParameterSet const* Event::parameterSet(edm::ParameterSetID const& psID) const {
    return parameterSetForID_(psID);
  }

  BasicHandle Event::getByProductID_(ProductID const& oid) const { return eventPrincipal().getByProductID(oid); }

  void Event::commit_(std::vector<edm::ProductResolverIndex> const& iShouldPut, ParentageID* previousParentageId) {
    size_t nPut = 0;
    for (auto const& p : putProducts()) {
      if (p) {
        ++nPut;
      }
    }
    if (nPut > 0) {
      commit_aux(putProducts(), previousParentageId);
    }
    auto sz = iShouldPut.size();
    if (sz != 0 and sz != nPut) {
      //some were missed
      auto& p = provRecorder_.principal();
      for (auto index : iShouldPut) {
        auto resolver = p.getProductResolverByIndex(index);
        if (not resolver->productResolved()) {
          resolver->putProduct(std::unique_ptr<WrapperBase>());
        }
      }
    }
  }

  void Event::commit_aux(Event::ProductPtrVec& products, ParentageID* previousParentageId) {
    // fill in guts of provenance here
    auto& ep = eventPrincipal();

    //If we don't have a valid previousParentage then we want to use a temp value in order to
    // avoid constantly recalculating the ParentageID which is a time consuming operation
    ParentageID const* presentParentageId;

    if (previousBranchIDs_) {
      bool sameAsPrevious = gotBranchIDs_.empty();
      if (sameAsPrevious) {
        for (auto i : gotBranchIDsFromPrevious_) {
          if (not i) {
            sameAsPrevious = false;
            break;
          }
        }
      }
      if (not sameAsPrevious) {
        std::vector<BranchID> gotBranchIDVector{gotBranchIDs_.begin(), gotBranchIDs_.end()};
        //add items in common from previous
        auto n = gotBranchIDsFromPrevious_.size();
        for (size_t i = 0; i < n; ++i) {
          if (gotBranchIDsFromPrevious_[i]) {
            gotBranchIDVector.push_back((*previousBranchIDs_)[i]);
          }
        }
        std::sort(gotBranchIDVector.begin(), gotBranchIDVector.end());
        previousBranchIDs_->assign(gotBranchIDVector.begin(), gotBranchIDVector.end());

        Parentage p;
        p.setParents(std::move(gotBranchIDVector));
        *previousParentageId = p.id();
        ParentageRegistry::instance()->insertMapped(p);
      }
      presentParentageId = previousParentageId;
    } else {
      presentParentageId = &s_emptyParentage;
    }

    auto const& recordProv = provRecorder_.recordProvenanceList();
    for (unsigned int i = 0; i < products.size(); ++i) {
      auto& p = get_underlying_safe(products[i]);
      if (p) {
        if (recordProv[i]) {
          ep.put(provRecorder_.putTokenIndexToProductResolverIndex()[i], std::move(p), *presentParentageId);
        } else {
          ep.put(provRecorder_.putTokenIndexToProductResolverIndex()[i], std::move(p), s_emptyParentage);
        }
      }
    }

    // the cleanup is all or none
    products.clear();
  }

  void Event::addToGotBranchIDs(Provenance const& prov) const { addToGotBranchIDs(prov.originalBranchID()); }

  void Event::addToGotBranchIDs(BranchID const& branchID) const {
    if (previousBranchIDs_) {
      auto range = std::equal_range(previousBranchIDs_->begin(), previousBranchIDs_->end(), branchID);
      if (range.first == range.second) {
        gotBranchIDs_.insert(branchID.id());
      } else {
        gotBranchIDsFromPrevious_[range.first - previousBranchIDs_->begin()] = true;
      }
    } else if (gotBranchIDsFromAcquire_) {
      gotBranchIDsFromAcquire_->push_back(branchID);
    }
  }

  ProcessHistory const& Event::processHistory() const { return provRecorder_.processHistory(); }

  size_t Event::size() const {
    return std::count_if(putProducts().begin(), putProducts().end(), [](auto const& i) { return bool(i); }) +
           provRecorder_.principal().size();
  }

  BasicHandle Event::getByLabelImpl(std::type_info const&,
                                    std::type_info const& iProductType,
                                    const InputTag& iTag) const {
    BasicHandle h = provRecorder_.getByLabel_(TypeID(iProductType), iTag, moduleCallingContext_);
    if (h.isValid()) {
      addToGotBranchIDs(*(h.provenance()));
    }
    return h;
  }

  BasicHandle Event::getImpl(std::type_info const&, ProductID const& pid) const {
    BasicHandle h = this->getByProductID_(pid);
    if (h.isValid()) {
      addToGotBranchIDs(*(h.provenance()));
    }
    return h;
  }

  TriggerNames const& Event::triggerNames(edm::TriggerResults const& triggerResults) const {
    edm::TriggerNames const* names = triggerNames_(triggerResults);
    if (names != nullptr)
      return *names;

    throw cms::Exception("TriggerNamesNotFound") << "TriggerNames not found in ParameterSet registry";
    return *names;
  }

  TriggerResultsByName Event::triggerResultsByName(edm::TriggerResults const& triggerResults) const {
    edm::TriggerNames const* names = triggerNames_(triggerResults);
    return TriggerResultsByName(&triggerResults, names);
  }
}  // namespace edm
