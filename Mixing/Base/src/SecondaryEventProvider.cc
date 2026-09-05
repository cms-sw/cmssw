#include "FWCore/Concurrency/interface/include_first_syncWait.h"
#include "Mixing/Base/src/SecondaryEventProvider.h"
#include "FWCore/Common/interface/ProcessBlockHelper.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/interface/ExceptionHelpers.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/SignallingProductRegistryFiller.h"
#include "FWCore/Framework/interface/ModuleRegistry.h"
#include "FWCore/Framework/interface/ModuleRegistryUtilities.h"
#include "FWCore/Framework/interface/maker/MakeModuleParams.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/make_sentry.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "oneapi/tbb/task_arena.h"

#include <mutex>

namespace {
  template <typename T, typename M, typename U>
  void processOneOccurrence(M& manager,
                            typename T::TransitionInfoType& info,
                            edm::StreamID streamID,
                            typename T::Context const* topContext,
                            U const* context,
                            bool cleaningUpAfterException = false) {
    manager.resetAll();

    if (manager.allWorkers().empty())
      return;

    auto token = edm::ServiceRegistry::instance().presentToken();
    //we need the arena to guarantee that the syncWait will return to this thread
    // and not cause this callstack to possibly be moved to a new thread
    tbb::task_arena localArena{tbb::this_task_arena::max_concurrency()};
    std::exception_ptr exceptPtr = localArena.execute([&]() {
      return edm::syncWait([&](edm::WaitingTaskHolder&& iHolder) {
        manager.template processOneOccurrenceAsync<T, U>(
            std::move(iHolder), info, token, streamID, topContext, context);
      });
    });

    if (exceptPtr) {
      try {
        edm::convertException::wrap([&]() { std::rethrow_exception(exceptPtr); });
      } catch (cms::Exception& ex) {
        if (ex.context().empty()) {
          edm::addContextAndPrintException("Calling SecondaryEventProvider", ex, cleaningUpAfterException);
        } else {
          edm::addContextAndPrintException("", ex, cleaningUpAfterException);
        }
        throw;
      }
    }
  }
}  // namespace

namespace edm {
  SecondaryEventProvider::SecondaryEventProvider(std::vector<ParameterSet>& psets,
                                                 SignallingProductRegistryFiller& preg,
                                                 std::shared_ptr<ProcessConfiguration> processConfiguration)
      : exceptionToActionTable_(new ExceptionToActionTable),
        moduleRegistry_(std::make_shared<ModuleRegistry>(nullptr)),
        activityRegistry_(std::make_shared<ActivityRegistry>()),
        // no type resolver for modules in SecondaryEventProvider for now
        runWorkerManager_(moduleRegistry_, activityRegistry_, *exceptionToActionTable_),
        runStreamWorkerManager_(moduleRegistry_, activityRegistry_, *exceptionToActionTable_),
        lumiWorkerManager_(moduleRegistry_, activityRegistry_, *exceptionToActionTable_),
        lumiStreamWorkerManager_(moduleRegistry_, activityRegistry_, *exceptionToActionTable_),
        eventWorkerManager_(moduleRegistry_, activityRegistry_, *exceptionToActionTable_) {
    std::vector<std::string> shouldBeUsedLabels;
    std::set<std::string> unscheduledLabels;
    const PreallocationConfiguration preallocConfig;
    for (auto& pset : psets) {
      std::string label = pset.getParameter<std::string>("@module_label");
      MakeModuleParams params(&pset, preg, &preallocConfig, processConfiguration);
      auto module = moduleRegistry_->getModule(params,
                                               label,
                                               activityRegistry_->preModuleConstructionSignal_,
                                               activityRegistry_->postModuleConstructionSignal_);
      if (module->moduleType() != edm::maker::ModuleHolder::Type::kProducer or
          module->moduleType() != edm::maker::ModuleHolder::Type::kFilter) {
        throw edm::Exception(edm::errors::Configuration)
            << "The module with label " << label << " is not an EDProducer or EDFilter so can not be run unscheduled";
      }
      eventWorkerManager_.addToUnscheduledWorkers(module->moduleDescription());
      if (module->wantsTransition(RunTransitionInfo::key(), TransitionPhaseGlobal::value)) {
        runWorkerManager_.addToUnscheduledWorkers(module->moduleDescription());
      }
      if (module->wantsTransition(RunTransitionInfo::key(), TransitionPhaseStream::value)) {
        runStreamWorkerManager_.addToUnscheduledWorkers(module->moduleDescription());
      }
      if (module->wantsTransition(LumiTransitionInfo::key(), TransitionPhaseGlobal::value)) {
        lumiWorkerManager_.addToUnscheduledWorkers(module->moduleDescription());
      }
      if (module->wantsTransition(LumiTransitionInfo::key(), TransitionPhaseStream::value)) {
        lumiStreamWorkerManager_.addToUnscheduledWorkers(module->moduleDescription());
      }

      unscheduledLabels.insert(label);
    }
    if (!unscheduledLabels.empty()) {
      preg.setUnscheduledProducts(unscheduledLabels);
    }
  }  // SecondaryEventProvider::SecondaryEventProvider

  void SecondaryEventProvider::beginJob(ProductRegistry const& iRegistry,
                                        eventsetup::ESRecordsToProductResolverIndices const& iIndices,
                                        GlobalContext const& globalContext) {
    ProcessBlockHelper dummyProcessBlockHelper;
    finishModulesInitialization(*moduleRegistry_,
                                iRegistry,
                                iIndices,
                                dummyProcessBlockHelper,
                                globalContext.processContext()->processConfiguration()->processName());
    runBeginJobForModules(globalContext, *moduleRegistry_, *activityRegistry_, modulesThatFailed_);
  }

  //NOTE: When the Stream interfaces are propagated to the modules, this code must be updated
  // to also send the stream based transitions
  void SecondaryEventProvider::beginRun(RunPrincipal& run,
                                        const EventSetupImpl& setup,
                                        ModuleCallingContext const* mcc,
                                        StreamContext& sContext) {
    RunTransitionInfo info(run, setup);
    processOneOccurrence<OccurrenceTraits<RunPrincipal, TransitionActionGlobalBegin>>(
        runWorkerManager_, info, StreamID::invalidStreamID(), nullptr, mcc);
    processOneOccurrence<OccurrenceTraits<RunPrincipal, TransitionActionStreamBegin>>(
        runStreamWorkerManager_, info, sContext.streamID(), &sContext, mcc);
  }

  void SecondaryEventProvider::beginLuminosityBlock(LuminosityBlockPrincipal& lumi,
                                                    const EventSetupImpl& setup,
                                                    ModuleCallingContext const* mcc,
                                                    StreamContext& sContext) {
    LumiTransitionInfo info(lumi, setup);
    processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionGlobalBegin>>(
        lumiWorkerManager_, info, StreamID::invalidStreamID(), nullptr, mcc);
    processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionStreamBegin>>(
        lumiStreamWorkerManager_, info, sContext.streamID(), &sContext, mcc);
  }

  void SecondaryEventProvider::endRun(RunPrincipal& run,
                                      const EventSetupImpl& setup,
                                      ModuleCallingContext const* mcc,
                                      StreamContext& sContext) {
    RunTransitionInfo info(run, setup);
    processOneOccurrence<OccurrenceTraits<RunPrincipal, TransitionActionStreamEnd>>(
        runStreamWorkerManager_, info, sContext.streamID(), &sContext, mcc);
    processOneOccurrence<OccurrenceTraits<RunPrincipal, TransitionActionGlobalEnd>>(
        runWorkerManager_, info, StreamID::invalidStreamID(), nullptr, mcc);
  }

  void SecondaryEventProvider::endLuminosityBlock(LuminosityBlockPrincipal& lumi,
                                                  const EventSetupImpl& setup,
                                                  ModuleCallingContext const* mcc,
                                                  StreamContext& sContext) {
    LumiTransitionInfo info(lumi, setup);
    processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionStreamEnd>>(
        lumiStreamWorkerManager_, info, sContext.streamID(), &sContext, mcc);
    processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionGlobalEnd>>(
        lumiWorkerManager_, info, StreamID::invalidStreamID(), nullptr, mcc);
  }

  void SecondaryEventProvider::setupPileUpEvent(EventPrincipal& ep,
                                                const EventSetupImpl& setup,
                                                StreamContext& sContext) {
    eventWorkerManager_.setupResolvers(ep);
    EventTransitionInfo info(ep, setup);
    eventWorkerManager_.setupOnDemandSystem(info);

    if (eventWorkerManager_.unscheduledWorkers().empty()) {
      return;
    }
    auto token = edm::ServiceRegistry::instance().presentToken();
    //we need the arena to guarantee that the syncWait will return to this thread
    // and not cause this callstack to possibly be moved to a new thread
    ParentContext pc(&sContext);
    std::exception_ptr exceptPtr = tbb::this_task_arena::isolate([&]() {
      return edm::syncWait([&](edm::WaitingTaskHolder&& iHolder) {
        for (auto& worker : eventWorkerManager_.unscheduledWorkers()) {
          worker->doWorkAsync<OccurrenceTraits<EventPrincipal, TransitionActionStreamBegin>>(
              iHolder, info, token, sContext.streamID(), pc, &sContext);
        }
      });
    });
    if (exceptPtr) {
      try {
        edm::convertException::wrap([&]() { std::rethrow_exception(exceptPtr); });
      } catch (cms::Exception& ex) {
        if (ex.context().empty()) {
          edm::addContextAndPrintException("Calling SecondaryEventProvider", ex, false);
        } else {
          edm::addContextAndPrintException("", ex, false);
        }
        throw;
      }
    }
  }

  void SecondaryEventProvider::beginStream(edm::StreamID iID, StreamContext const& sContext) {
    //can reuse modulesThatFailed_ since we can't get here of failed in beginRun
    runBeginStreamForModules(sContext, *moduleRegistry_, *activityRegistry_, modulesThatFailed_);
  }

  void SecondaryEventProvider::endStream(edm::StreamID iID,
                                         StreamContext const& sContext,
                                         ExceptionCollector& exceptionCollector) {
    // In this context the mutex is not needed because these things are not
    // executing concurrently but in general the WorkerManager needs one.
    std::mutex exceptionCollectorMutex;
    //modulesThatFailed_ gets used in endJob and we can only get here if endJob succeeded
    auto sentry = make_sentry(&modulesThatFailed_, [](auto* failed) { failed->clear(); });
    runEndStreamForModules(
        sContext, *moduleRegistry_, *activityRegistry_, exceptionCollector, exceptionCollectorMutex, modulesThatFailed_);
  }

  void SecondaryEventProvider::endJob(ExceptionCollector& exceptionCollector, GlobalContext const& globalContext) {
    runEndJobForModules(globalContext, *moduleRegistry_, *activityRegistry_, exceptionCollector, modulesThatFailed_);
  }

}  // namespace edm
