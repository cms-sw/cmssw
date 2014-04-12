#include "Mixing/Base/src/SecondaryEventProvider.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

namespace edm {
  SecondaryEventProvider::SecondaryEventProvider(std::vector<ParameterSet>& psets,
                     ProductRegistry& preg,
                     boost::shared_ptr<ProcessConfiguration> processConfiguration) :
    exceptionToActionTable_(new ExceptionToActionTable),
    workerManager_(boost::shared_ptr<ActivityRegistry>(new ActivityRegistry), *exceptionToActionTable_) {
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
                                               false,
                                               unscheduledLabels,
                                               shouldBeUsedLabels);
    }
    if(!unscheduledLabels.empty()) {
      workerManager_.setOnDemandProducts(preg, unscheduledLabels); 
    }
  } // SecondaryEventProvider::SecondaryEventProvider
  
  //NOTE: When the Stream interfaces are propagated to the modules, this code must be updated
  // to also send the stream based transitions
  void SecondaryEventProvider::beginRun(RunPrincipal& run, const EventSetup& setup, ModuleCallingContext const* mcc) {
    workerManager_.processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin> >(run, setup, StreamID::invalidStreamID(),
                                                                                                  mcc->getGlobalContext(), mcc);
  }

  void SecondaryEventProvider::beginLuminosityBlock(LuminosityBlockPrincipal& lumi, const EventSetup& setup, ModuleCallingContext const* mcc) {
    workerManager_.processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin> >(lumi, setup, StreamID::invalidStreamID(),
                                                                                                              mcc->getGlobalContext(), mcc);
  }

  void SecondaryEventProvider::endRun(RunPrincipal& run, const EventSetup& setup, ModuleCallingContext const* mcc) {
    workerManager_.processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd> >(run, setup, StreamID::invalidStreamID(),
                                                                                                mcc->getGlobalContext(), mcc);
  }

  void SecondaryEventProvider::endLuminosityBlock(LuminosityBlockPrincipal& lumi, const EventSetup& setup, ModuleCallingContext const* mcc) {
    workerManager_.processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd> >(lumi, setup, StreamID::invalidStreamID(),
                                                                                                            mcc->getGlobalContext(), mcc);
  }

  void SecondaryEventProvider::setupPileUpEvent(EventPrincipal& ep, const EventSetup& setup) {
    workerManager_.processOneOccurrence<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin>, StreamContext >(ep, setup, ep.streamID(),
                                                                                                                   nullptr, nullptr);
  }
}
