/*----------------------------------------------------------------------
  
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProducerBase.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/ProducesCollector.h"

#include <sstream>

namespace edm {
  ProducerBase::ProducerBase() : ProductRegistryHelper(), callWhenNewProductsRegistered_() {}
  ProducerBase::~ProducerBase() noexcept(false) {}

  std::function<void(BranchDescription const&)> ProducerBase::registrationCallback() const {
    return callWhenNewProductsRegistered_;
  }

  namespace {
    class CallbackWrapper {
    public:
      CallbackWrapper(ProductRegistryHelper* iProd,
                      std::function<void(BranchDescription const&)> iCallback,
                      ProductRegistry* iReg,
                      const ModuleDescription& iDesc)
          : prod_(iProd), callback_(iCallback), reg_(iReg), mdesc_(iDesc), lastSize_(iProd->typeLabelList().size()) {}

      void operator()(BranchDescription const& iDesc) {
        callback_(iDesc);
        addToRegistry();
      }

      void addToRegistry() {
        ProducerBase::TypeLabelList const& plist = prod_->typeLabelList();

        if (lastSize_ != plist.size()) {
          ProducerBase::TypeLabelList::const_iterator pStart = plist.begin();
          advance(pStart, lastSize_);
          ProductRegistryHelper::addToRegistry(pStart, plist.end(), mdesc_, *reg_, prod_);
          lastSize_ = plist.size();
        }
      }

    private:
      ProductRegistryHelper* prod_;
      std::function<void(BranchDescription const&)> callback_;
      ProductRegistry* reg_;
      ModuleDescription mdesc_;
      unsigned int lastSize_;
    };
  }  // namespace

  void ProducerBase::registerProducts(ProducerBase* producer, ProductRegistry* iReg, ModuleDescription const& md) {
    if (typeLabelList().empty() && !registrationCallback()) {
      return;
    }
    //If we have a callback, first tell the callback about all the entries already in the
    // product registry, then add any items this producer wants to add to the registry
    // and only after that do we register the callback. This is done so the callback does not
    // get called for items registered by this producer (avoids circular reference problems)
    bool isListener = false;
    if (registrationCallback()) {
      isListener = true;
      iReg->callForEachBranch(registrationCallback());
    }
    TypeLabelList const& plist = typeLabelList();

    ProductRegistryHelper::addToRegistry(plist.begin(), plist.end(), md, *(iReg), this, isListener);
    if (registrationCallback()) {
      Service<ConstProductRegistry> regService;
      regService->watchProductAdditions(CallbackWrapper(producer, registrationCallback(), iReg, md));
    }
  }

  void ProducerBase::resolvePutIndicies(BranchType iBranchType,
                                        ModuleToResolverIndicies const& iIndicies,
                                        std::string const& moduleLabel) {
    auto const& plist = typeLabelList();
    if (putTokenToResolverIndex_.size() != plist.size()) {
      putTokenToResolverIndex_.resize(plist.size(), std::numeric_limits<unsigned int>::max());
    }
    auto range = iIndicies.equal_range(moduleLabel);
    putIndicies_[iBranchType].reserve(iIndicies.count(moduleLabel));
    for (auto it = range.first; it != range.second; ++it) {
      putIndicies_[iBranchType].push_back(std::get<2>(it->second));
      unsigned int i = 0;
      for (auto const& tl : plist) {
        if (convertToBranchType(tl.transition_) == iBranchType and (tl.typeID_ == *std::get<0>(it->second)) and
            (tl.productInstanceName_ == std::get<1>(it->second))) {
          putTokenToResolverIndex_[i] = std::get<2>(it->second);
          //NOTE: The ExternalLHEProducer puts the 'same' product in at
          // both beginRun and endRun. Therefore we can get multiple matches.
          // When the module is finally fixed, we can use the 'break'
          //break;
        }
        ++i;
      }
    }
  }

  ProducesCollector ProducerBase::producesCollector() { return ProducesCollector{this}; }

}  // namespace edm
