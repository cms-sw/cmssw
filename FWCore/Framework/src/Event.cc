#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/StableProvenance.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {

  std::string const Event::emptyString_;

  Event::Event(EventPrincipal const& ep, ModuleDescription const& md, ModuleCallingContext const* moduleCallingContext) :
      provRecorder_(ep, md),
      aux_(ep.aux()),
      luminosityBlock_(ep.luminosityBlockPrincipalPtrValid() ? new LuminosityBlock(ep.luminosityBlockPrincipal(), md, moduleCallingContext) : nullptr),
      gotBranchIDs_(),
      gotViews_(),
      streamID_(ep.streamID()),
      moduleCallingContext_(moduleCallingContext)
  {
  }

  Event::~Event() {
  }

  Event::CacheIdentifier_t
  Event::cacheIdentifier() const {
    return eventPrincipal().cacheIdentifier();
  }

  void
  Event::setConsumer(EDConsumerBase const* iConsumer) {
    provRecorder_.setConsumer(iConsumer);
    const_cast<LuminosityBlock*>(luminosityBlock_.get())->setConsumer(iConsumer);
  }
  
  void
  Event::setSharedResourcesAcquirer( SharedResourcesAcquirer* iResourceAcquirer) {
    provRecorder_.setSharedResourcesAcquirer(iResourceAcquirer);
    const_cast<LuminosityBlock*>(luminosityBlock_.get())->setSharedResourcesAcquirer(iResourceAcquirer);
  }


  EventPrincipal const&
  Event::eventPrincipal() const {
    return dynamic_cast<EventPrincipal const&>(provRecorder_.principal());
  }

  EDProductGetter const&
  Event::productGetter() const {
    return provRecorder_.principal();
  }

  ProductID
  Event::makeProductID(BranchDescription const& desc) const {
    return eventPrincipal().branchIDToProductID(desc.originalBranchID());
  }

  Run const&
  Event::getRun() const {
    return getLuminosityBlock().getRun();
  }

  EventSelectionIDVector const&
  Event::eventSelectionIDs() const {
    return eventPrincipal().eventSelectionIDs();
  }

  ProcessHistoryID const&
  Event::processHistoryID() const {
    return eventPrincipal().processHistoryID();
  }

  Provenance
  Event::getProvenance(BranchID const& bid) const {
    return provRecorder_.principal().getProvenance(bid, moduleCallingContext_);
  }

  Provenance
  Event::getProvenance(ProductID const& pid) const {
    return eventPrincipal().getProvenance(pid, moduleCallingContext_);
  }

  void
  Event::getAllProvenance(std::vector<Provenance const*>& provenances) const {
    provRecorder_.principal().getAllProvenance(provenances);
  }

  void
  Event::getAllStableProvenance(std::vector<StableProvenance const*>& provenances) const {
    provRecorder_.principal().getAllStableProvenance(provenances);
  }

  bool
  Event::getProcessParameterSet(std::string const& processName,
                                ParameterSet& ps) const {
    ProcessConfiguration config;
    bool process_found = processHistory().getConfigurationForProcess(processName, config);
    if(process_found) {
      pset::Registry::instance()->getMapped(config.parameterSetID(), ps);
      assert(!ps.empty());
    }
    return process_found;
  }

  BasicHandle
  Event::getByProductID_(ProductID const& oid) const {
    return eventPrincipal().getByProductID(oid);
  }

  void
  Event::commit_(std::vector<edm::ProductResolverIndex> const& iShouldPut,
                 std::vector<BranchID>* previousParentage, ParentageID* previousParentageId) {
    auto nPut = putProducts().size()+putProductsWithoutParents().size();
    commit_aux(putProducts(), true, previousParentage, previousParentageId);
    commit_aux(putProductsWithoutParents(), false);
    auto sz = iShouldPut.size();
    if(sz !=0 and sz != nPut) {
      //some were missed
      auto& p = provRecorder_.principal();
      for(auto index: iShouldPut){
        auto resolver = p.getProductResolverByIndex(index);
        if(not resolver->productResolved()) {
          resolver->putProduct(std::unique_ptr<WrapperBase>());
        }
      }
    }
  }

  void
  Event::commit_aux(Event::ProductPtrVec& products, bool record_parents,
                    std::vector<BranchID>* previousParentage, ParentageID* previousParentageId) {
    // fill in guts of provenance here
    auto& ep = eventPrincipal();

    ProductPtrVec::iterator pit(products.begin());
    ProductPtrVec::iterator pie(products.end());

    std::vector<BranchID> gotBranchIDVector;

    // Note that gotBranchIDVector will remain empty if
    // record_parents is false (and may be empty if record_parents is
    // true).

    //Check that previousParentageId is still good by seeing if previousParentage matches gotBranchIDs_
    bool sameAsPrevious = ((0 != previousParentage) && (previousParentage->size() == gotBranchIDs_.size()));
    if(record_parents && !gotBranchIDs_.empty()) {
      gotBranchIDVector.reserve(gotBranchIDs_.size());
      std::vector<BranchID>::const_iterator itPrevious;
      if(previousParentage) {
        itPrevious = previousParentage->begin();
      }
      for(BranchIDSet::const_iterator it = gotBranchIDs_.begin(), itEnd = gotBranchIDs_.end();
          it != itEnd; ++it) {
        gotBranchIDVector.push_back(*it);
        if(sameAsPrevious) {
          if(*it != *itPrevious) {
            sameAsPrevious = false;
          } else {
            ++itPrevious;
          }
        }
      }
      if(!sameAsPrevious && 0 != previousParentage) {
        previousParentage->assign(gotBranchIDVector.begin(), gotBranchIDVector.end());
      }
    }

    //If we don't have a valid previousParentage then we want to use a temp value in order to
    // avoid constantly recalculating the ParentageID which is a time consuming operation
    ParentageID temp;
    if(!previousParentage) {
      assert(!sameAsPrevious);
      previousParentageId = &temp;
    }
    while(pit != pie) {
      // set provenance
      if(!sameAsPrevious) {
        ProductProvenance prov(pit->second->branchID(), std::move(gotBranchIDVector));
        *previousParentageId = prov.parentageID();
        ep.put(*pit->second, std::move(get_underlying_safe(pit->first)), prov);
        sameAsPrevious = true;
      } else {
        ProductProvenance prov(pit->second->branchID(), *previousParentageId);
        ep.put(*pit->second, std::move(get_underlying_safe(pit->first)), prov);
      }
      ++pit;
    }

    // the cleanup is all or none
    products.clear();
  }

  void
  Event::addToGotBranchIDs(Provenance const& prov) const {
    gotBranchIDs_.insert(prov.originalBranchID());
  }

  ProcessHistory const&
  Event::processHistory() const {
    return provRecorder_.processHistory();
  }

  size_t
  Event::size() const {
    return putProducts().size() + provRecorder_.principal().size() + putProductsWithoutParents().size();
  }

  BasicHandle
  Event::getByLabelImpl(std::type_info const&, std::type_info const& iProductType, const InputTag& iTag) const {
    BasicHandle h = provRecorder_.getByLabel_(TypeID(iProductType), iTag, moduleCallingContext_);
    if(h.isValid()) {
      addToGotBranchIDs(*(h.provenance()));
    }
    return h;
  }

  BasicHandle
  Event::getImpl(std::type_info const&, ProductID const& pid) const {
    BasicHandle h = this->getByProductID_(pid);
    if(h.isValid()) {
      addToGotBranchIDs(*(h.provenance()));
    }
    return h;
  }

  TriggerNames const&
  Event::triggerNames(edm::TriggerResults const& triggerResults) const {
    edm::TriggerNames const* names = triggerNames_(triggerResults);
    if(names != 0) return *names;

    throw cms::Exception("TriggerNamesNotFound")
      << "TriggerNames not found in ParameterSet registry";
    return *names;
  }

  TriggerResultsByName
  Event::triggerResultsByName(edm::TriggerResults const& triggerResults) const {

    edm::TriggerNames const* names = triggerNames_(triggerResults);
    return TriggerResultsByName(&triggerResults, names);
  }
}
