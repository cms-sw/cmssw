# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Reconstruction targets: offline vs Phase-2 HLT.

The same builder declaration can be emitted for different targets.  A ``Target``
captures everything that differs between them: the module label scheme, the
merged layer-cluster source (and the other shared inputs derived from it), the
shared module labels (seeding / tile / links / candidate / pf), and whether the
stages are grouped into ``cms.Task`` (offline) or ``cms.Sequence`` (HLT).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import FWCore.ParameterSet.Config as cms


@dataclass(frozen=True)
class Target:
    name: str
    trackster_prefix: str
    filter_prefix: str
    seeding_labels: Dict[str, str]
    layer_tile_label: str
    links_label: str
    supercluster_dnn_label: str
    egamma_label: str
    candidate_label: str
    mtd_label: str
    pf_label: str
    grouping: str                       # 'Task' or 'Sequence'
    # names of the assembled groups (Tasks/Sequences); 'step' is a template
    group_names: Dict[str, str] = field(default_factory=dict)
    # merged layer-cluster module; None -> keep the cfi defaults (offline)
    merged_lc: Optional[str] = None
    # InputTag for layer_clusters_barrel_tiles; None -> keep the cfi default
    barrel_tile_tag: Optional[object] = None

    # -- group naming ------------------------------------------------------ #

    def gname(self, role, name=None):
        tmpl = self.group_names[role]
        return tmpl.format(name=name) if name is not None else tmpl

    # -- label helpers ----------------------------------------------------- #

    def trackster_label(self, name):
        return self.trackster_prefix + name

    def filter_label(self, name):
        return self.filter_prefix + name

    def seeding_label(self, seeding_type):
        return self.seeding_labels.get(seeding_type)

    # -- shared input tags derived from the merged layer clusters ---------- #

    def lc_tag(self):
        return cms.InputTag(self.merged_lc) if self.merged_lc else None

    def lc_time_tag(self):
        return cms.InputTag(self.merged_lc, "timeLayerCluster") if self.merged_lc else None

    def initial_mask_tag(self):
        return cms.InputTag(self.merged_lc, "InitialLayerClustersMask") if self.merged_lc else None

    def initial_mask_str(self):
        # the "module:instance" string form, as the baseline renders VInputTags
        return "%s:InitialLayerClustersMask" % self.merged_lc if self.merged_lc else None


OFFLINE = Target(
    name="offline",
    trackster_prefix="ticlTracksters",
    filter_prefix="filteredLayerClusters",
    seeding_labels={"SeedingRegionGlobal": "ticlSeedingGlobal",
                    "SeedingRegionByTracks": "ticlSeedingTrk"},
    layer_tile_label="ticlLayerTileProducer",
    links_label="ticlTracksterLinks",
    supercluster_dnn_label="ticlTracksterLinksSuperclusteringDNN",
    egamma_label="ticlEGammaSuperClusterProducer",
    candidate_label="ticlCandidate",
    mtd_label="mtdSoA",
    pf_label="pfTICL",
    grouping="Task",
    group_names={
        "layer_tile": "ticlLayerTileTask",
        "step": "ticl{name}StepTask",
        "iterations": "ticlIterationsTask",
        "superclustering": "ticlSuperclusteringTask",
        "links": "ticlTracksterLinksTask",
        "merge": "mergeTICLTask",
        "mtd": "mtdSoATask",
        "candidate": "ticlCandidateTask",
        "pf": "ticlPFTask",
        "top": "iterTICLTask",
    },
    merged_lc=None,
    barrel_tile_tag=None,
)


HLT = Target(
    name="HLT",
    trackster_prefix="hltTiclTracksters",
    filter_prefix="hltFilteredLayerClusters",
    seeding_labels={"SeedingRegionGlobal": "hltTiclSeedingGlobal",
                    "SeedingRegionByL1": "hltTiclSeedingL1"},
    layer_tile_label="hltTiclLayerTileProducer",
    links_label="hltTiclTracksterLinks",
    supercluster_dnn_label="hltTiclTracksterLinksSuperclusteringDNN",
    egamma_label="hltTiclEGammaSuperClusterProducer",
    candidate_label="hltTiclCandidate",
    mtd_label="hltMtdSoA",
    pf_label="hltPfTICL",
    grouping="Sequence",
    group_names={
        "layer_tile": "HLTTiclLayerTileSequence",
        "step": "HLTTiclTracksters{name}StepSequence",
        "iterations": "HLTTiclIterationsSequence",
        "superclustering": "HLTTiclSuperclusteringSequence",
        "links": "HLTTiclTracksterLinksSequence",
        "merge": "HLTMergeTICLSequence",
        "mtd": "HLTMtdSoASequence",
        "candidate": "HLTTiclCandidateSequence",
        "pf": "HLTTiclPFSequence",
        "top": "HLTIterTICLSequence",
    },
    merged_lc="hltMergeLayerClusters",
    barrel_tile_tag=cms.InputTag("hltTiclLayerTileBarrelProducer", "ticlLayerTilesBarrel"),
)


TARGETS = {"offline": OFFLINE, "HLT": HLT}
