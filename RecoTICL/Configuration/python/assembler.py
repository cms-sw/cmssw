# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Assemble a :class:`~RecoTICL.Configuration.model.TICLConfig` into ``cms`` objects.

Each node is built by cloning the *real* ``_cfi`` default (so values match the
baseline) and applying (a) the algorithm overrides recorded in the model and
(b) the *plumbing* ``InputTag``s computed here from the iteration graph.  The
result is a set of labelled modules plus the nested ``Task`` structure
(``ticl<Iter>StepTask`` -> ``ticlIterationsTask`` -> ``mergeTICLTask`` ->
``iterTICLTask``) matching ``iterativeTICL_cff``.
"""

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration.catalog import CATALOG, SEEDING_MODULE_LABEL
from RecoTICL.Configuration.model import PyTICLError


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _cfi_default(key):
    spec = CATALOG[key]
    mod = __import__(spec.cfi_module, fromlist=[spec.cfi_symbol])
    return getattr(mod, spec.cfi_symbol)


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
    """The product of assembling a config: labelled modules + named tasks."""

    def __init__(self, config):
        self.config = config
        self.modules = {}        # label -> cms module   (insertion order preserved)
        self.tasks = {}          # name  -> cms.Task
        self.task_children = {}  # name  -> [child identifier, ...] (labels/task names)
        self.top = None          # the iterTICLTask

    def add_to_process(self, process):
        """Register every module (labelled) and task on ``process``."""
        for label, mod in self.modules.items():
            setattr(process, label, mod)
        for name, task in self.tasks.items():
            setattr(process, name, task)
        return process


# --------------------------------------------------------------------------- #
# per-node builders
# --------------------------------------------------------------------------- #

def _build_seeding(seeding_type):
    base = _cfi_default("TICLSeedingRegionProducer")
    return base.clone(seedingPSet=base.seedingPSet.clone(type=seeding_type))


def _build_layer_tile():
    return _cfi_default("TICLLayerTileProducer").clone()


def _build_filter(it, prev_trackster):
    base = _cfi_default("FilteredLayerClustersProducer")
    ov = dict(clusterFilter=it.filter_type, iteration_label=it.name)
    ov.update(it.filter_params)
    if prev_trackster:
        ov["LayerClustersInputMask"] = cms.InputTag(prev_trackster)
    return base.clone(**ov)


def _build_trackster(it, flabel, seed_label, prev_trackster):
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
    ov.update(it.trackster_extra)
    return base.clone(**ov)


def _build_links(collection_labels, overrides):
    base = _cfi_default("TracksterLinksProducer")
    # baseline builds this as cms.VInputTag(*string_labels) -- reproduce exactly
    ov = dict(tracksters_collections=cms.VInputTag(*collection_labels))
    ov.update(overrides or {})
    return base.clone(**ov)


def _build_supercluster_dnn(source_label, overrides):
    base = _cfi_default("TracksterLinksProducer")
    # baseline builds this as a python list of cms.InputTag -- reproduce exactly
    ov = dict(tracksters_collections=[cms.InputTag(source_label)])
    ov.update(overrides or {})
    return base.clone(**ov)


def _build_egamma():
    return _cfi_default("EGammaSuperclusterProducer").clone()


def _build_candidate(overrides):
    return _cfi_default("TICLCandidateProducer").clone(**(overrides or {}))


def _build_mtd():
    return _cfi_default("MTDSoAProducer").clone()


def _build_pf(overrides):
    base = _cfi_default("PFTICLProducer")
    ov = dict(ticlCandidateSrc=cms.InputTag("ticlCandidate"))
    ov.update(overrides or {})
    return base.clone(**ov)


# --------------------------------------------------------------------------- #
# top-level assembly
# --------------------------------------------------------------------------- #

def assemble(cfg):
    res = Assembled(cfg)
    m = res.modules
    t = res.tasks

    def mktask(name, *children):
        """Create ``cms.Task(name)`` from child identifiers (module labels or
        task names already built), recording the composition for export."""
        objs = []
        for ch in children:
            if ch in m:
                objs.append(m[ch])
            elif ch in t:
                objs.append(t[ch])
            else:
                raise PyTICLError("task %r references unknown child %r" % (name, ch))
        t[name] = cms.Task(*objs)
        res.task_children[name] = list(children)
        return t[name]

    def ensure_seeding(stype):
        label = SEEDING_MODULE_LABEL.get(stype)
        if label is None:
            raise PyTICLError("no canonical seeding module for seeding type %r; "
                              "extend catalog.SEEDING_MODULE_LABEL" % stype)
        if label not in m:
            m[label] = _build_seeding(stype)
        return label

    # shared infrastructure: layer tile
    if cfg.include_layer_tile:
        m["ticlLayerTileProducer"] = _build_layer_tile()
        mktask("ticlLayerTileTask", "ticlLayerTileProducer")

    # iterations
    step_tasks = []
    for it in cfg.iterations:
        if it.seeding_type is None:
            raise PyTICLError("iteration %r has no seeding region" % it.name)
        if it.filter_type is None:
            raise PyTICLError("iteration %r has no cluster filter" % it.name)
        if it.pattern_type is None:
            raise PyTICLError("iteration %r has no pattern recognition" % it.name)
        seed_label = ensure_seeding(it.seeding_type)
        flabel = filter_label(it.name)
        tlabel = trackster_label(it.name)
        prev = None
        if it.masks_from:
            if it.masks_from not in cfg._by_name:
                raise PyTICLError("iteration %r masks_from unknown iteration %r"
                                  % (it.name, it.masks_from))
            prev = trackster_label(it.masks_from)
        m[flabel] = _build_filter(it, prev)
        m[tlabel] = _build_trackster(it, flabel, seed_label, prev)
        sname = step_task_name(it.name)
        mktask(sname, seed_label, flabel, tlabel)
        step_tasks.append(sname)

    if step_tasks:
        mktask("ticlIterationsTask", *step_tasks)

    # links + superclustering
    if cfg.links_spec:
        labels = [trackster_label(n) for n in cfg.links_spec.collections]
        m["ticlTracksterLinks"] = _build_links(labels, cfg.links_spec.overrides)

    if cfg.superclustering_spec:
        sc = cfg.superclustering_spec
        src = trackster_label(sc.source)
        m["ticlTracksterLinksSuperclusteringDNN"] = _build_supercluster_dnn(src, sc.overrides)
        m["ticlEGammaSuperClusterProducer"] = _build_egamma()
        mktask("ticlSuperclusteringTask",
               "ticlTracksterLinksSuperclusteringDNN", "ticlEGammaSuperClusterProducer")

    if "ticlTracksterLinks" in m:
        links_children = ["ticlTracksterLinks"]
        if "ticlSuperclusteringTask" in t:
            links_children.append("ticlSuperclusteringTask")
        mktask("ticlTracksterLinksTask", *links_children)

    # mergeTICLTask
    merge = [name for name in ("ticlLayerTileTask", "ticlIterationsTask",
                               "ticlTracksterLinksTask") if name in t]
    mktask("mergeTICLTask", *merge)

    # candidate / mtd / pf
    if cfg.include_mtd:
        m["mtdSoA"] = _build_mtd()
        mktask("mtdSoATask", "mtdSoA")
    if cfg.include_candidate:
        m["ticlCandidate"] = _build_candidate(cfg.candidate_spec)
        mktask("ticlCandidateTask", "ticlCandidate")
    if cfg.include_pf:
        m["pfTICL"] = _build_pf(cfg.pf_spec)
        mktask("ticlPFTask", "pfTICL")

    # iterTICLTask
    top = ["mergeTICLTask"] + [name for name in ("mtdSoATask", "ticlCandidateTask",
                                                 "ticlPFTask") if name in t]
    mktask("iterTICLTask", *top)
    res.top = t["iterTICLTask"]
    return res
