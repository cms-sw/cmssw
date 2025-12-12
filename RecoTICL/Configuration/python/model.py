# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""The pyTICL fluent builder -- the user-facing API.

A ``TICLConfig`` records *intent* only (no ``cms`` objects).  Iterations are
described with chained calls; top-level stages (links, superclustering,
candidate, pf, validation) follow.  ``assemble()`` / ``validate()`` / ``to_cff()``
turn the intent into real ``cms`` modules, check the plumbing, and export.

Example::

    from RecoTICL.Configuration.model import TICLConfig, Global
    cfg = (TICLConfig('v5')
           .iteration('CLUE3DHigh')
               .seeding(Global).filter_by_algo_and_size(min_size=2)
               .pattern_clue3d(criticalDensity=[0.6, 0.6, 0.6])
           .iteration('Recovery').preset().masks_from('CLUE3DHigh')
           .links(['CLUE3DHigh', 'Recovery'])
           .superclustering_dnn(source='CLUE3DHigh')
           .candidate().pf())
    cfg.validate()
    cfg.to_cff('myTICL_cff.py')
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from RecoTICL.Configuration.catalog import SEEDING_TYPES, CPU, GPU
from RecoTICL.Configuration.target import OFFLINE, TARGETS


class PyTICLError(Exception):
    """Raised for invalid pyTICL configurations (bad plumbing, unknown types...)."""


# --------------------------------------------------------------------------- #
# Symbolic constants for the fluent API
# --------------------------------------------------------------------------- #

Global = "SeedingRegionGlobal"
ByTracks = "SeedingRegionByTracks"
ByHF = "SeedingRegionByHF"
ByL1 = "SeedingRegionByL1"


# --------------------------------------------------------------------------- #
# Intent dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class IterationSpec:
    """One TICL iteration: seeding region + cluster filter + pattern recognition."""
    name: str
    seeding_type: Optional[str] = None
    filter_type: Optional[str] = None
    filter_params: Dict = field(default_factory=dict)
    pattern_type: Optional[str] = None
    pattern_params: Dict = field(default_factory=dict)
    trackster_extra: Dict = field(default_factory=dict)  # inference, detector, ...
    masks_from: Optional[str] = None                     # name of the upstream iteration
    backend: str = CPU
    detector: str = "HGCAL"


@dataclass
class LinksSpec:
    collections: List[str]            # iteration names feeding ticlTracksterLinks
    overrides: Dict = field(default_factory=dict)


@dataclass
class SuperclusterSpec:
    kind: str                         # 'DNN' or 'Mustache'
    source: str                       # iteration name (or module label) to supercluster
    overrides: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# The builder
# --------------------------------------------------------------------------- #

class TICLConfig:
    def __init__(self, name="pyticl", target=OFFLINE):
        self.name = name
        self.target = TARGETS[target] if isinstance(target, str) else target
        self.iterations: List[IterationSpec] = []
        self._by_name: Dict[str, IterationSpec] = {}
        self._current: Optional[IterationSpec] = None
        self.links_spec: Optional[LinksSpec] = None
        self.superclustering_spec: Optional[SuperclusterSpec] = None
        self.candidate_spec: Optional[Dict] = None
        self.pf_spec: Optional[Dict] = None
        self.include_layer_tile = True
        self.include_mtd = True
        self.include_candidate = False
        self.include_pf = False
        self.validation_spec: Optional[Dict] = None

    # -- iteration context ------------------------------------------------- #

    def iteration(self, name):
        if name in self._by_name:
            raise PyTICLError("duplicate iteration name: %r" % name)
        it = IterationSpec(name=name)
        self.iterations.append(it)
        self._by_name[name] = it
        self._current = it
        return self

    def _cur(self):
        if self._current is None:
            raise PyTICLError("no current iteration -- call .iteration(name) first")
        return self._current

    # -- seeding ----------------------------------------------------------- #

    def seeding(self, seeding_type):
        if seeding_type not in SEEDING_TYPES:
            raise PyTICLError("unknown seeding type %r (known: %s)"
                              % (seeding_type, ", ".join(sorted(SEEDING_TYPES))))
        self._cur().seeding_type = seeding_type
        return self

    # -- cluster filters --------------------------------------------------- #

    def _set_filter(self, ftype, **params):
        it = self._cur()
        it.filter_type = ftype
        it.filter_params = {k: v for k, v in params.items() if v is not None}
        return self

    def filter_by_algo(self, algo_number=None, min_layerId=None, max_layerId=None):
        return self._set_filter("ClusterFilterByAlgo", algo_number=algo_number,
                                min_layerId=min_layerId, max_layerId=max_layerId)

    def filter_by_size(self, min_size=None, max_size=None):
        return self._set_filter("ClusterFilterBySize",
                                min_cluster_size=min_size, max_cluster_size=max_size)

    def filter_by_algo_and_size(self, min_size=None, max_size=None, algo_number=None):
        return self._set_filter("ClusterFilterByAlgoAndSize", min_cluster_size=min_size,
                                max_cluster_size=max_size, algo_number=algo_number)

    def filter_by_algo_and_size_and_layer_range(self, min_size=None, max_size=None,
                                                 algo_number=None, min_layerId=None,
                                                 max_layerId=None):
        return self._set_filter("ClusterFilterByAlgoAndSizeAndLayerRange",
                                min_cluster_size=min_size, max_cluster_size=max_size,
                                algo_number=algo_number, min_layerId=min_layerId,
                                max_layerId=max_layerId)

    # -- pattern recognition ---------------------------------------------- #

    def pattern_clue3d(self, **params):
        it = self._cur(); it.pattern_type = "CLUE3D"; it.pattern_params = params; return self

    def pattern_ca(self, **params):
        it = self._cur(); it.pattern_type = "CA"; it.pattern_params = params; return self

    def pattern_recovery(self, **params):
        it = self._cur(); it.pattern_type = "Recovery"; it.pattern_params = params; return self

    def pattern_fastjet(self, **params):
        it = self._cur(); it.pattern_type = "FastJet"; it.pattern_params = params; return self

    # -- extra trackster overrides (inference PSets, detector, ...) -------- #

    def trackster_params(self, **overrides):
        self._cur().trackster_extra.update(overrides)
        return self

    def masks_from(self, prev_name):
        self._cur().masks_from = prev_name
        return self

    # -- backend ----------------------------------------------------------- #

    def on_cpu(self):
        self._cur().backend = CPU
        return self

    def on_gpu(self):
        self._cur().backend = GPU
        return self

    # -- presets ----------------------------------------------------------- #

    def preset(self, name=None):
        from RecoTICL.Configuration import presets
        presets.apply_iteration_preset(self._cur(), name or self._cur().name)
        return self

    # -- top-level stages -------------------------------------------------- #

    def links(self, collections, **overrides):
        self.links_spec = LinksSpec(collections=list(collections), overrides=overrides)
        self._current = None
        return self

    def superclustering_dnn(self, source, **overrides):
        self.superclustering_spec = SuperclusterSpec("DNN", source, overrides)
        self._current = None
        return self

    def candidate(self, **overrides):
        self.include_candidate = True
        self.candidate_spec = overrides
        self._current = None
        return self

    def pf(self, **overrides):
        self.include_pf = True
        self.pf_spec = overrides
        self._current = None
        return self

    def validation(self, dumper=False, **opts):
        self.validation_spec = dict(dumper=dumper, **opts)
        self._current = None
        return self

    # -- terminal operations ---------------------------------------------- #

    def assemble(self):
        from RecoTICL.Configuration.assembler import assemble
        return assemble(self)

    def validate(self):
        from RecoTICL.Configuration.validator import validate
        return validate(self)

    def to_cff(self, path, process_name="TICL"):
        from RecoTICL.Configuration.exporter import to_cff
        return to_cff(self, path, process_name)
