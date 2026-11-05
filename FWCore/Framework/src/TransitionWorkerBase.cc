
/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "FWCore/Concurrency/interface/include_first_syncWait.h"
#include "FWCore/Framework/interface/maker/TransitionWorkerBase.h"
#include "FWCore/Framework/interface/EarlyDeleteHelper.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/src/EventAcquireSignalsSentry.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/ParameterSet/interface/Registry.h"

namespace edm {

  TransitionWorkerBase::TransitionWorkerBase(ModuleDescription const& iMD, ExceptionToActionTable const* iActions)
      : state_(Ready),
        moduleCallingContext_(&iMD),
        actions_(iActions),
        cached_exception_(),
        actReg_(),
        workStarted_(false) {
    checkForShouldTryToContinue(iMD);
  }

  TransitionWorkerBase::~TransitionWorkerBase() {}

  void TransitionWorkerBase::setActivityRegistry(std::shared_ptr<ActivityRegistry> areg) { actReg_ = areg; }

  void TransitionWorkerBase::checkForShouldTryToContinue(ModuleDescription const& iDesc) {
    auto pset = edm::pset::Registry::instance()->getMapped(iDesc.parameterSetID());
    if (pset and pset->exists("@shouldTryToContinue")) {
      shouldTryToContinue_ = true;
    }
  }

  bool TransitionWorkerBase::shouldRethrowException(std::exception_ptr iPtr,
                                                    ParentContext const& parentContext,
                                                    bool isEvent,
                                                    bool shouldTryToContinue) const noexcept {
    // NOTE: the warning printed as a result of ignoring or failing
    // a module will only be printed during the full true processing
    // pass of this module

    // Get the action corresponding to this exception.  However, if processing
    // something other than an event (e.g. run, lumi) always rethrow.
    if (not isEvent) {
      return true;
    }
    try {
      convertException::wrap([&]() { std::rethrow_exception(iPtr); });
    } catch (cms::Exception& ex) {
      exception_actions::ActionCodes action = actions_->find(ex.category());

      if (action == exception_actions::Rethrow) {
        return true;
      }
      if (action == exception_actions::TryToContinue) {
        if (shouldTryToContinue) {
          edm::printCmsExceptionWarning("TryToContinue", ex);
        }
        return not shouldTryToContinue;
      }
      if (action == exception_actions::IgnoreCompletely) {
        edm::printCmsExceptionWarning("IgnoreCompletely", ex);
        return false;
      }
    }
    return true;
  }

  void TransitionWorkerBase::esPrefetchAsync(WaitingTaskHolder iTask,
                                             EventSetupImpl const& iImpl,
                                             Transition iTrans,
                                             ServiceToken const& iToken) noexcept {
    if (iTrans >= edm::Transition::NumberOfEventSetupTransitions) {
      return;
    }
    auto const& recs = esRecordsToGetFrom(iTrans);
    auto const& items = esItemsToGetFrom(iTrans);

    assert(items.size() == recs.size());
    if (items.empty()) {
      return;
    }

    for (size_t i = 0; i != items.size(); ++i) {
      if (recs[i] != ESRecordIndex{}) {
        auto rec = iImpl.findImpl(recs[i]);
        if (rec) {
          rec->prefetchAsync(iTask, items[i], &iImpl, iToken, ESParentContext(&moduleCallingContext_));
        }
      }
    }
  }

  void TransitionWorkerBase::edPrefetchAsync(WaitingTaskHolder iTask,
                                             ServiceToken const& token,
                                             Principal const& iPrincipal) const noexcept {
    // Prefetch products the module declares it consumes
    std::vector<ProductResolverIndexAndSkipBit> const& items = itemsToGetFrom(iPrincipal.branchType());

    for (auto const& item : items) {
      ProductResolverIndex productResolverIndex = item.productResolverIndex();
      if (productResolverIndex != ProductResolverIndexAmbiguous) {
        iPrincipal.prefetchAsync(iTask, productResolverIndex, token, &moduleCallingContext_);
      }
    }
  }
  void TransitionWorkerBase::resetModuleDescription(ModuleDescription const* iDesc) {
    ModuleCallingContext temp(iDesc,
                              0,
                              moduleCallingContext_.state(),
                              moduleCallingContext_.parent(),
                              moduleCallingContext_.previousModuleOnThread());
    moduleCallingContext_ = temp;
    assert(iDesc);
    checkForShouldTryToContinue(*iDesc);
  }
}  // namespace edm
