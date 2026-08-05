# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# Part of the MC-truth-graph prototype - under heavy development, not yet open
# to external contributions (see PhysicsTools/TruthInfo/README.md).

"""RECO-step customise for the pileup truth chain in a split production.

The stage-independent truth (logical graph + UNRESOLVED per-particle per-cell hit
index + raw graph TruthGraph_mix) is built at the DIGI step by
mixedTruthGraphCustomize.customiseTruthDigi, where the merged signal+pileup simHits
are live, and arrives here through the DIGI-RAW input file. This step:

  1. drops the signal-only truth (re)build that the enableTruth validation would
     otherwise schedule at RECO (truthGraphProducer, truthLogicalGraphProducer,
     detIdToRecHitMapProducer, truthLogicalGraphHitIndexProducer). With those modules
     gone, the branch validators' and association producers' empty-process InputTags
     resolve to the mixed products from the DIGI-RAW input instead of a signal-only
     rebuild;
  2. repoints the branch validators' rawSrc to the mixed raw graph (TruthGraph_mix,
     label 'mix'); the association producers carry no rawSrc;
  3. re-persists the truth into the RECO-tier output at the requested verbosity level.

No rebuild and no simHits are needed here: the shared-energy association
(BranchHitAssociator) matches by DetId, so the unresolved index is sufficient, and
no validator or associator reads recHitIndex."""

import FWCore.ParameterSet.Config as cms

from PhysicsTools.TruthInfo.truthEventContent_cff import setTruthEventContent

# The signal-only truth (re)build modules that enableTruth would schedule at RECO.
# In the mixed production these are already built at DIGI, so they must not run here.
_signalOnlyBuildModules = [
    'truthGraphProducer',
    'truthLogicalGraphProducer',
    'detIdToRecHitMapProducer',
    'truthLogicalGraphHitIndexProducer',
]

# Branch validators that read the raw graph via rawSrc (the association producers do
# not); repointed to the mixed raw graph.
_rawSrcConsumers = [
    'branchHGCalValidator',
    'branchTrackingValidator',
]


def _detachModule(process, name):
    """Remove a module from every sequence/path/endpath/task and detach it from the
    process, so it is neither scheduled nor an on-demand current-process product and
    its label resolves to the input file's product."""
    if not hasattr(process, name):
        return
    module = getattr(process, name)
    containers = []
    for accessor in ('sequences', 'paths', 'endpaths', 'tasks', 'finalpaths'):
        holder = getattr(process, accessor, None)
        if holder is not None:
            containers.extend(holder.values())
    for container in containers:
        try:
            container.remove(module)
        except Exception:
            pass
    delattr(process, name)


def customiseValidation(process):
    """Point the RECO-side enableTruth validation at the DIGI-built mixed truth."""
    for name in _signalOnlyBuildModules:
        _detachModule(process, name)
    for name in _rawSrcConsumers:
        if hasattr(process, name):
            getattr(process, name).rawSrc = cms.InputTag('mix')
    return process


def customiseContent(process, level='compact', includeTrackingHits=True):
    """Persist the mixed truth into the RECO-tier output at the given level."""
    return setTruthEventContent(process, level=level, includeTrackingHits=includeTrackingHits)


def customise(process, level='compact', includeTrackingHits=True):
    """RECO-step entry point: repoint validation to the mixed truth and persist it."""
    process = customiseValidation(process)
    process = customiseContent(process, level=level, includeTrackingHits=includeTrackingHits)
    return process
