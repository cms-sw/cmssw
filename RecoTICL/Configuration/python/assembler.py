# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Assemble a :class:`~RecoTICL.Configuration.model.TICLConfig` into ``cms`` objects.

Each node is built by cloning the *real* ``_cfi`` default (so values match the
baseline) and applying (a) the algorithm overrides recorded in the model and
(b) the *plumbing* ``InputTag``s computed here from the iteration graph and the
selected :class:`~RecoTICL.Configuration.target.Target` (offline or HLT).  The result
is a set of labelled modules plus the grouping (``cms.Task`` offline /
``cms.Sequence`` HLT).
"""

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration.catalog import CATALOG
from RecoTICL.Configuration.model import PyTICLError


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _cfi_default(key):
    spec = CATALOG[key]
    mod = __import__(spec.cfi_module, fromlist=[spec.cfi_symbol])
    return getattr(mod, spec.cfi_symbol)


# offline default label helpers (used by the validator and by callers that
# pre-date targets); the target's own helpers are preferred inside assembly.
def trackster_label(name):
    return "ticlTracksters" + name


def filter_label(name):
    return "filteredLayerClusters" + name


def step_task_name(name):
    return "ticl" + name + "StepTask"


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #

class Assembled:
    """The product of assembling a config: labelled modules + named groups."""

    def __init__(self, config):
        self.config = config
        self.target = config.target
        self.modules = {}        # label -> cms module   (insertion order preserved)
        self.tasks = {}          # name  -> cms.Task / cms.Sequence
        self.task_children = {}  # name  -> [child identifier, ...]
        self.top = None          # the top group (iterTICLTask / HLTIterTICLSequence)

    def add_to_process(self, process):
        for label, mod in self.modules.items():
            setattr(process, label, mod)
        for name, group in self.tasks.items():
            setattr(process, name, group)
        return process


# --------------------------------------------------------------------------- #
# per-node builders (target-aware)
# --------------------------------------------------------------------------- #

def _build_seeding(seeding_type):
    base = _cfi_default("TICLSeedingRegionProducer")
    return base.clone(seedingPSet=base.seedingPSet.clone(type=seeding_type))


def _build_layer_tile(target):
    ov = {}
    if target.lc_tag() is not None:
        ov["layer_clusters"] = target.lc_tag()
    return _cfi_default("TICLLayerTileProducer").clone(**ov)


def _build_filter(it, prev_trackster, target):
    base = _cfi_default("FilteredLayerClustersProducer")
    ov = dict(clusterFilter=it.filter_type, iteration_label=it.name)
    ov.update(it.filter_params)
    if target.lc_tag() is not None:
        ov["LayerClusters"] = target.lc_tag()
    if prev_trackster:
        ov["LayerClustersInputMask"] = cms.InputTag(prev_trackster)
    elif target.initial_mask_tag() is not None:
        ov["LayerClustersInputMask"] = target.initial_mask_tag()
    return base.clone(**ov)


def _build_trackster(it, flabel, seed_label, prev_trackster, target):
    base = _cfi_default("TrackstersProducer")
    ov = dict(
        filtered_mask=cms.InputTag(flabel, it.name),
        seeding_regions=cms.InputTag(seed_label),
        itername=it.name,
        patternRecognitionBy=it.pattern_type,
    )
    ov["pluginPatternRecognitionBy" + it.pattern_type] = dict(**it.pattern_params)
    if prev_trackster:
        ov["original_mask"] = cms.InputTag(prev_trackster)
    if target.lc_tag() is not None:  # HLT-style: remap the shared inputs
        ov["layer_clusters"] = target.lc_tag()
        ov["time_layerclusters"] = target.lc_time_tag()
        ov["layer_clusters_tiles"] = cms.InputTag(target.layer_tile_label)
        if target.barrel_tile_tag is not None:
            ov["layer_clusters_barrel_tiles"] = target.barrel_tile_tag
        if not prev_trackster:
            ov["original_mask"] = target.initial_mask_tag()
    ov.update(it.trackster_extra)
    return base.clone(**ov)


def _build_links(collection_labels, overrides, target):
    base = _cfi_default("TracksterLinksProducer")
    ov = dict(tracksters_collections=cms.VInputTag(*collection_labels))
    if target.lc_tag() is not None:
        ov["layer_clusters"] = target.lc_tag()
        ov["layer_clustersTime"] = target.lc_time_tag()
        ov["original_masks"] = cms.VInputTag(target.initial_mask_str())
    ov.update(overrides or {})
    return base.clone(**ov)


def _build_supercluster_dnn(source_label, overrides, target):
    base = _cfi_default("TracksterLinksProducer")
    ov = dict(tracksters_collections=[cms.InputTag(source_label)])
    if target.lc_tag() is not None:
        ov["layer_clusters"] = target.lc_tag()
        ov["layer_clustersTime"] = target.lc_time_tag()
        ov["original_masks"] = cms.VInputTag(target.initial_mask_str())
    ov.update(overrides or {})
    return base.clone(**ov)


def _build_egamma(target):
    return _cfi_default("EGammaSuperclusterProducer").clone()


def _build_candidate(overrides, links_label, target):
    base = _cfi_default("TICLCandidateProducer")
    ov = {}
    if target.lc_tag() is not None:  # HLT-style: remap shared inputs + link collections
        ov["layer_clusters"] = target.lc_tag()
        ov["layer_clustersTime"] = target.lc_time_tag()
        ov["original_masks"] = cms.VInputTag(target.initial_mask_str())
        link_tag = cms.VInputTag(links_label)
        ov["egamma_tracksters_collections"] = link_tag
        ov["egamma_tracksterlinks_collections"] = cms.VInputTag(links_label)
        ov["general_tracksters_collections"] = cms.VInputTag(links_label)
        ov["general_tracksterlinks_collections"] = cms.VInputTag(links_label)
    ov.update(overrides or {})
    return base.clone(**ov)


def _build_mtd():
    return _cfi_default("MTDSoAProducer").clone()


def _build_pf(overrides, target):
    base = _cfi_default("PFTICLProducer")
    ov = dict(ticlCandidateSrc=cms.InputTag(target.candidate_label))
    ov.update(overrides or {})
    return base.clone(**ov)


# --------------------------------------------------------------------------- #
# top-level assembly
# --------------------------------------------------------------------------- #

def assemble(cfg):
    res = Assembled(cfg)
    target = cfg.target
    m = res.modules
    t = res.tasks

    def group(*items):
        if target.grouping == "Sequence":
            combined = items[0]
            for nxt in items[1:]:
                combined = combined + nxt
            return cms.Sequence(combined)
        return cms.Task(*items)

    def mkgroup(name, *children):
        objs = []
        for ch in children:
            if ch in m:
                objs.append(m[ch])
            elif ch in t:
                objs.append(t[ch])
            else:
                raise PyTICLError("group %r references unknown child %r" % (name, ch))
        t[name] = group(*objs)
        res.task_children[name] = list(children)
        return t[name]

    def ensure_seeding(stype):
        label = target.seeding_label(stype)
        if label is None:
            raise PyTICLError("target %r has no seeding module for seeding type %r"
                              % (target.name, stype))
        if label not in m:
            m[label] = _build_seeding(stype)
        return label

    # shared infrastructure: layer tile
    if cfg.include_layer_tile:
        m[target.layer_tile_label] = _build_layer_tile(target)
        mkgroup(target.gname("layer_tile"), target.layer_tile_label)

    # iterations
    step_groups = []
    for it in cfg.iterations:
        if it.seeding_type is None:
            raise PyTICLError("iteration %r has no seeding region" % it.name)
        if it.filter_type is None:
            raise PyTICLError("iteration %r has no cluster filter" % it.name)
        if it.pattern_type is None:
            raise PyTICLError("iteration %r has no pattern recognition" % it.name)
        seed_label = ensure_seeding(it.seeding_type)
        flabel = target.filter_label(it.name)
        tlabel = target.trackster_label(it.name)
        prev = None
        if it.masks_from:
            if it.masks_from not in cfg._by_name:
                raise PyTICLError("iteration %r masks_from unknown iteration %r"
                                  % (it.name, it.masks_from))
            prev = target.trackster_label(it.masks_from)
        m[flabel] = _build_filter(it, prev, target)
        m[tlabel] = _build_trackster(it, flabel, seed_label, prev, target)
        gname = target.gname("step", it.name)
        mkgroup(gname, seed_label, flabel, tlabel)
        step_groups.append(gname)

    if step_groups:
        mkgroup(target.gname("iterations"), *step_groups)

    # links + superclustering
    if cfg.links_spec:
        labels = [target.trackster_label(n) for n in cfg.links_spec.collections]
        m[target.links_label] = _build_links(labels, cfg.links_spec.overrides, target)

    if cfg.superclustering_spec:
        sc = cfg.superclustering_spec
        src = target.trackster_label(sc.source)
        m[target.supercluster_dnn_label] = _build_supercluster_dnn(src, sc.overrides, target)
        m[target.egamma_label] = _build_egamma(target)
        mkgroup(target.gname("superclustering"),
                target.supercluster_dnn_label, target.egamma_label)

    if target.links_label in m:
        links_children = [target.links_label]
        if target.gname("superclustering") in t:
            links_children.append(target.gname("superclustering"))
        mkgroup(target.gname("links"), *links_children)

    # mergeTICL
    merge = [target.gname(r) for r in ("layer_tile", "iterations", "links")
             if target.gname(r) in t]
    mkgroup(target.gname("merge"), *merge)

    # candidate / mtd / pf
    if cfg.include_mtd:
        m[target.mtd_label] = _build_mtd()
        mkgroup(target.gname("mtd"), target.mtd_label)
    if cfg.include_candidate:
        m[target.candidate_label] = _build_candidate(
            cfg.candidate_spec, target.links_label, target)
        mkgroup(target.gname("candidate"), target.candidate_label)
    if cfg.include_pf:
        m[target.pf_label] = _build_pf(cfg.pf_spec, target)
        mkgroup(target.gname("pf"), target.pf_label)

    # top group
    top = [target.gname("merge")] + [target.gname(r) for r in ("mtd", "candidate", "pf")
                                     if target.gname(r) in t]
    mkgroup(target.gname("top"), *top)
    res.top = t[target.gname("top")]
    return res
