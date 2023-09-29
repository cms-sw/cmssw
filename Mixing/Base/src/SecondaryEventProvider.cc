#include "FWCore/Concurrency/interface/include_first_syncWait.h"
#include "Mixing/Base/src/SecondaryEventProvider.h"
#include "FWCore/Common/interface/ProcessBlockHelper.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "oneapi/tbb/task_arena.h"

namespace {
  template <typename T, typename U>
  void processOneOccurrence(edm::WorkerManager& manager,
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
        manager.processOneOccurrenceAsync<T, U>(std::move(iHolder), info, token, streamID, topContext, context);
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
                                                 ProductRegistry& preg,
                                                 std::shared_ptr<ProcessConfiguration> processConfiguration)
      : exceptionToActionTable_(new ExceptionToActionTable),
        // no type resolver for modules in SecondaryEventProvider for now
        workerManager_(std::make_shared<ActivityRegistry>(), *exceptionToActionTable_, nullptr) {
    std::vector<std::string> shouldBeUsedLabels;
    std::set<std::string> unscheduledLabels;
    const PreallocationConfiguration preallocConfig;
    for (auto& pset : psets) {
      std::string label = pset.getParameter<std::string>("@module_label");
      workerManager_.addToUnscheduledWorkers(
          pset, preg, &preallocConfig, processConfiguration, label, unscheduledLabels, shouldBeUsedLabels);
    }
    if (!unscheduledLabels.empty()) {
      preg.setUnscheduledProducts(unscheduledLabels);
    }
  }  // SecondaryEventProvider::SecondaryEventProvider

  void SecondaryEventProvider::beginJob(ProductRegistry const& iRegistry,
                                        eventsetup::ESRecordsToProductResolverIndices const& iIndices) {
    ProcessBlockHelper dummyProcessBlockHelper;
    workerManager_.beginJob(iRegistry, iIndices, dummyProcessBlockHelper);
  }

  //NOTE: When the Stream interfaces are propagated to the modules, this code must be updated
  // to also send the stream based transitions
  void SecondaryEventProvider::beginRun(RunPrincipal& run,
                                        const EventSetupImpl& setup,
                                        ModuleCallingContext const* mcc,
                                        StreamContext& sContext) {
    RunTransitionInfo info(run, setup);
    processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin>>(
        workerManager_, info, StreamID::invalidStreamID(), nullptr, mcc);
    processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionStreamBegin>>(
        workerManager_, info, sContext.streamID(), &sContext, mcc);
  }

  void SecondaryEventProvider::beginLuminosityBlock(LuminosityBlockPrincipal& lumi,
                                                    const EventSetupImpl& setup,
                                                    ModuleCallingContext const* mcc,
                                                    StreamContext& sContext) {
    LumiTransitionInfo info(lumi, setup);
    processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin>>(
        workerManager_, info, StreamID::invalidStreamID(), nullptr, mcc);
    processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin>>(
        workerManager_, info, sContext.streamID(), &sContext, mcc);
  }

  void SecondaryEventProvider::endRun(RunPrincipal& run,
                                      const EventSetupImpl& setup,
                                      ModuleCallingContext const* mcc,
                                      StreamContext& sContext) {
    RunTransitionInfo info(run, setup);
    processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionStreamEnd>>(
        workerManager_, info, sContext.streamID(), &sContext, mcc);
    processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd>>(
        workerManager_, info, StreamID::invalidStreamID(), nullptr, mcc);
  }

  void SecondaryEventProvider::endLuminosityBlock(LuminosityBlockPrincipal& lumi,
                                                  const EventSetupImpl& setup,
                                                  ModuleCallingContext const* mcc,
                                                  StreamContext& sContext) {
    LumiTransitionInfo info(lumi, setup);
    processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd>>(
        workerManager_, info, sContext.streamID(), &sContext, mcc);
    processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd>>(
        workerManager_, info, StreamID::invalidStreamID(), nullptr, mcc);
  }

  void SecondaryEventProvider::setupPileUpEvent(EventPrincipal& ep,
                                                const EventSetupImpl& setup,
                                                StreamContext& sContext) {
    workerManager_.setupResolvers(ep);
    EventTransitionInfo info(ep, setup);
    workerManager_.setupOnDemandSystem(info);

    if (workerManager_.unscheduledWorkers().empty()) {
      return;
    }
    auto token = edm::ServiceRegistry::instance().presentToken();
    //we need the arena to guarantee that the syncWait will return to this thread
    // and not cause this callstack to possibly be moved to a new thread
    ParentContext pc(&sContext);
    std::exception_ptr exceptPtr = tbb::this_task_arena::isolate([&]() {
      return edm::syncWait([&](edm::WaitingTaskHolder&& iHolder) {
        for (auto& worker : workerManager_.unscheduledWorkers()) {
          worker->doWorkAsync<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>>(
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

  void SecondaryEventProvider::beginStream(edm::StreamID iID, StreamContext& sContext) {
    workerManager_.beginStream(iID, sContext);
  }

  void SecondaryEventProvider::endStream(edm::StreamID iID, StreamContext& sContext) {
    workerManager_.endStream(iID, sContext);
  }
}  // namespace edm
