#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/WorkerManager.h"

namespace edm {

  void OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin>::setupOnDemandSystem(WorkerManager& workerManager,
                                                                                    RunTransitionInfo& transitionInfo) {
    workerManager.setupOnDemandSystem(transitionInfo.principal(), &transitionInfo.eventSetupImpl());
  }

  void OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd>::setupOnDemandSystem(WorkerManager& workerManager,
                                                                                  RunTransitionInfo& transitionInfo) {
    workerManager.setupOnDemandSystem(transitionInfo.principal(), &transitionInfo.eventSetupImpl());
  }

  void OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin>::setupOnDemandSystem(
      WorkerManager& workerManager, LumiTransitionInfo& transitionInfo) {
    workerManager.setupOnDemandSystem(transitionInfo.principal(), &transitionInfo.eventSetupImpl());
  }

  void OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd>::setupOnDemandSystem(
      WorkerManager& workerManager, LumiTransitionInfo& transitionInfo) {
    workerManager.setupOnDemandSystem(transitionInfo.principal(), &transitionInfo.eventSetupImpl());
  }

  void OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalBegin>::setupOnDemandSystem(
      WorkerManager& workerManager, ProcessBlockTransitionInfo& transitionInfo) {
    workerManager.setupOnDemandSystem(transitionInfo.principal());
  }

  void OccurrenceTraits<ProcessBlockPrincipal, BranchActionProcessBlockInput>::setupOnDemandSystem(
      WorkerManager& workerManager, ProcessBlockTransitionInfo& transitionInfo) {
    workerManager.setupOnDemandSystem(transitionInfo.principal());
  }

  void OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalEnd>::setupOnDemandSystem(
      WorkerManager& workerManager, ProcessBlockTransitionInfo& transitionInfo) {
    workerManager.setupOnDemandSystem(transitionInfo.principal());
  }

}  // namespace edm
