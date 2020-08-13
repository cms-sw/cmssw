#include "Mixing/Base/src/SecondaryEventProvider.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Framework/src/TransitionInfoTypes.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"

namespace edm {
  SecondaryEventProvider::SecondaryEventProvider(std::vector<ParameterSet>& psets,
                                                 ProductRegistry& preg,
                                                 std::shared_ptr<ProcessConfiguration> processConfiguration)
      : exceptionToActionTable_(new ExceptionToActionTable),
        workerManager_(std::make_shared<ActivityRegistry>(), *exceptionToActionTable_) {
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
                                        eventsetup::ESRecordsToProxyIndices const& iIndices) {
    workerManager_.beginJob(iRegistry, iIndices);
  }

  //NOTE: When the Stream interfaces are propagated to the modules, this code must be updated
  // to also send the stream based transitions
  void SecondaryEventProvider::beginRun(RunPrincipal& run,
                                        const EventSetupImpl& setup,
                                        ModuleCallingContext const* mcc,
                                        StreamContext& sContext) {
    RunTransitionInfo info(run, setup);
    workerManager_.processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin> >(
        info, StreamID::invalidStreamID(), nullptr, mcc);
    workerManager_.processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionStreamBegin> >(
        info, sContext.streamID(), &sContext, mcc);
  }

  void SecondaryEventProvider::beginLuminosityBlock(LuminosityBlockPrincipal& lumi,
                                                    const EventSetupImpl& setup,
                                                    ModuleCallingContext const* mcc,
                                                    StreamContext& sContext) {
    LumiTransitionInfo info(lumi, setup);
    workerManager_.processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin> >(
        info, StreamID::invalidStreamID(), nullptr, mcc);
    workerManager_.processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin> >(
        info, sContext.streamID(), &sContext, mcc);
  }

  void SecondaryEventProvider::endRun(RunPrincipal& run,
                                      const EventSetupImpl& setup,
                                      ModuleCallingContext const* mcc,
                                      StreamContext& sContext) {
    RunTransitionInfo info(run, setup);
    workerManager_.processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionStreamEnd> >(
        info, sContext.streamID(), &sContext, mcc);
    workerManager_.processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd> >(
        info, StreamID::invalidStreamID(), nullptr, mcc);
  }

  void SecondaryEventProvider::endLuminosityBlock(LuminosityBlockPrincipal& lumi,
                                                  const EventSetupImpl& setup,
                                                  ModuleCallingContext const* mcc,
                                                  StreamContext& sContext) {
    LumiTransitionInfo info(lumi, setup);
    workerManager_.processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd> >(
        info, sContext.streamID(), &sContext, mcc);
    workerManager_.processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd> >(
        info, StreamID::invalidStreamID(), nullptr, mcc);
  }

  void SecondaryEventProvider::setupPileUpEvent(EventPrincipal& ep,
                                                const EventSetupImpl& setup,
                                                StreamContext& sContext) {
    workerManager_.setupOnDemandSystem(ep);
    EventTransitionInfo info(ep, setup);
    workerManager_.setupOnDemandSystem(info);
  }
  void SecondaryEventProvider::beginStream(edm::StreamID iID, StreamContext& sContext) {
    workerManager_.beginStream(iID, sContext);
  }

  void SecondaryEventProvider::endStream(edm::StreamID iID, StreamContext& sContext) {
    workerManager_.endStream(iID, sContext);
  }
}  // namespace edm
