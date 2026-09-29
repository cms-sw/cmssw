# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Orchestrate HGCAL local reconstruction + layer clustering with TICL.

pyTICL does not re-derive the local reconstruction (rechit calibration, the CLUE
clustering plugins, ...); it *orchestrates* the existing modules.  This module
attaches the HGCAL ``hgcalLocalRecoTask`` (which produces ``hgcalMergeLayerClusters``)
upstream of a TICL configuration, registers the merged layer-cluster producer so
the type-aware validator checks the **local-reco -> TICL boundary** (TICL's
``layer_clusters`` / ``original_mask`` / ``time_layerclusters`` are satisfied by
the merge), and lets the whole chain be scheduled and exported together.

CPU/GPU: the HGCAL SoA rechit & clustering producers are portable; drive them
with :mod:`RecoTICL.Configuration.backend` (see M4).  The default offline layer
clustering is the CPU CLUE plugin.
"""

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration.validator import validate_assembled

MERGED_LAYER_CLUSTERS = "hgcalMergeLayerClusters"


def hgcal_local_reco_task():
    """The existing HGCAL local-reco Task (rechits -> layer clusters -> merge)."""
    from RecoLocalCalo.Configuration.hgcalLocalReco_cff import hgcalLocalRecoTask
    return hgcalLocalRecoTask


def merged_layer_clusters():
    """The configured ``hgcalMergeLayerClusters`` producer TICL consumes."""
    from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import hgcalMergeLayerClusters
    return hgcalMergeLayerClusters


def attach_local_reco(assembled, validate=True):
    """Register the merged layer-cluster producer on ``assembled`` so the validator
    type-checks the local-reco -> TICL boundary.  Returns the (mutated) result."""
    assembled.modules[MERGED_LAYER_CLUSTERS] = merged_layer_clusters()
    if validate:
        validate_assembled(assembled)
    return assembled


def build_full_reco(cfg, validate=True):
    """Assemble a TICL config and attach HGCAL local reco upstream.

    The returned :class:`Assembled` has ``hgcalMergeLayerClusters`` in its product
    graph, so the type-aware validator confirms TICL's layer-cluster inputs are
    satisfied (and would reject a type-incompatible source)."""
    return attach_local_reco(cfg.assemble(), validate=validate)


def add_to_process(process, assembled):
    """Schedule local reco + TICL on ``process`` and return a combined Task.

    Loads ``hgcalLocalReco_cff`` (which labels & schedules the local-reco modules
    and provides ``hgcalMergeLayerClusters``), registers the TICL modules/groups,
    and builds ``fullRecoTask = Task(hgcalLocalRecoTask, <ticl top>)``."""
    process.load("RecoLocalCalo.Configuration.hgcalLocalReco_cff")
    assembled.add_to_process(process)
    top = getattr(process, assembled.target.gname("top"))
    process.fullRecoTask = cms.Task(process.hgcalLocalRecoTask, top)
    return process.fullRecoTask
