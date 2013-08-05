#include "Mixing/Base/src/SecondaryEventProvider.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

namespace edm {
  SecondaryEventProvider::SecondaryEventProvider(std::vector<ParameterSet>& psets,
                     ProductRegistry& preg,
                     ActionTable const& actions,
                     boost::shared_ptr<ProcessConfiguration> processConfiguration) :
    workerManager_(boost::shared_ptr<ActivityRegistry>(new ActivityRegistry), actions) {
    std::vector<std::string> shouldBeUsedLabels;
    std::set<std::string> unscheduledLabels;
    for(auto& pset : psets) {
        std::string label = pset.getParameter<std::string>("@module_label");
        workerManager_.addToUnscheduledWorkers(pset,
                                               preg,
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
  void SecondaryEventProvider::beginRun(RunPrincipal& run, const EventSetup& setup) {
    workerManager_.processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin> >(run, setup, StreamID::invalidStreamID());
  }

  void SecondaryEventProvider::beginLuminosityBlock(LuminosityBlockPrincipal& lumi, const EventSetup& setup) {
    workerManager_.processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin> >(lumi, setup, StreamID::invalidStreamID());
  }

  void SecondaryEventProvider::endRun(RunPrincipal& run, const EventSetup& setup) {
    workerManager_.processOneOccurrence<OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd> >(run, setup, StreamID::invalidStreamID());
  }

  void SecondaryEventProvider::endLuminosityBlock(LuminosityBlockPrincipal& lumi, const EventSetup& setup) {
    workerManager_.processOneOccurrence<OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd> >(lumi, setup, StreamID::invalidStreamID());
  }

  void SecondaryEventProvider::setupPileUpEvent(EventPrincipal& ep, const EventSetup& setup) {
    workerManager_.processOneOccurrence<OccurrenceTraits<EventPrincipal, BranchActionStreamBegin> >(ep, setup, ep.streamID());
  }
}
