#include "Mixing/Base/src/SecondaryEventProvider.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

namespace edm {
  SecondaryEventProvider::SecondaryEventProvider(std::vector<ParameterSet>& psets,
                     ProductRegistry& preg,
                     std::shared_ptr<ProcessConfiguration> processConfiguration) :
    exceptionToActionTable_(new ExceptionToActionTable),
    workerManager_(std::make_shared<ActivityRegistry>(), *exceptionToActionTable_) {
    std::vector<std::string> shouldBeUsedLabels;
    std::set<std::string> unscheduledLabels;
    const PreallocationConfiguration preallocConfig;
    for(auto& pset : psets) {
        std::string label = pset.getParameter<std::string>("@module_label");
        workerManager_.addToUnscheduledWorkers(pset,
                                               preg,
                                               &preallocConfig,
                                               processConfiguration,
                                               label,
                                               unscheduledLabels,
                                               shouldBeUsedLabels);
    }
    if(!unscheduledLabels.empty()) {
      workerManager_.setOnDemandProducts(preg, unscheduledLabels);
    }
  } // SecondaryEventProvider::SecondaryEventProvider

  //NOTE: When the Stream interfaces are propagated to the modules, this code must be updated
  // to also send the stream based transitions
  void SecondaryEventProvider::beginRun(RunPrincipal& run, const EventSetup& setup, ModuleCallingContext const* mcc, StreamContext& sContext) {
    workerManager_.processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin> >(run, setup, StreamID::invalidStreamID(),
                                                                                                  nullptr, mcc);
    workerManager_.processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionStreamBegin> >(run, setup, sContext.streamID(),
                                                                                                  &sContext, mcc);
  }

  void SecondaryEventProvider::beginLuminosityBlock(LuminosityBlockPrincipal& lumi, const EventSetup& setup, ModuleCallingContext const* mcc, StreamContext& sContext) {
    workerManager_.processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin> >(lumi, setup, StreamID::invalidStreamID(),
                                                                                                              nullptr, mcc);
    workerManager_.processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin> >(lumi, setup, sContext.streamID(),
                                                                                                              &sContext, mcc);
  }

  void SecondaryEventProvider::endRun(RunPrincipal& run, const EventSetup& setup, ModuleCallingContext const* mcc, StreamContext& sContext) {
    workerManager_.processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionStreamEnd> >(run, setup, sContext.streamID(),
                                                                                                &sContext, mcc);
    workerManager_.processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd> >(run, setup, StreamID::invalidStreamID(),
                                                                                                nullptr, mcc);
  }

  void SecondaryEventProvider::endLuminosityBlock(LuminosityBlockPrincipal& lumi, const EventSetup& setup, ModuleCallingContext const* mcc, StreamContext& sContext) {
    workerManager_.processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd> >(lumi, setup, sContext.streamID(),
                                                                                                            &sContext, mcc);
    workerManager_.processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd> >(lumi, setup, StreamID::invalidStreamID(),
                                                                                                            nullptr, mcc);
  }

  void SecondaryEventProvider::setupPileUpEvent(EventPrincipal& ep, const EventSetup& setup, StreamContext& sContext) {
    workerManager_.setupOnDemandSystem(ep, setup);
  }
  void SecondaryEventProvider::beginStream(edm::StreamID iID, StreamContext& sContext) {
    workerManager_.beginStream(iID, sContext);
  }

  void SecondaryEventProvider::endStream(edm::StreamID iID, StreamContext& sContext) {
    workerManager_.endStream(iID, sContext);
  }
}
