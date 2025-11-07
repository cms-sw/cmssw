# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Type-aware module registry for pyTICL.

This is pyTICL's "type system". For every TICL producer it records:

* ``cfi``        -- where to import the real default module from, so the
                    assembler can clone it (guaranteeing the generated config
                    matches the baseline byte-for-byte);
* ``produces``   -- the C++ products the module puts into the event, each with
                    an *instance-label rule* so the validator knows the exact
                    ``InputTag(module, instance)`` that resolves to it;
* ``consumes``   -- every ``InputTag`` / ``VInputTag`` parameter the module
                    reads, with the required C++ product type, so the validator
                    can reject type-incompatible connections;
* ``backends``   -- which compute backends the module supports (``cpu`` only, or
                    ``cpu`` + ``gpu``/alpaka).

The data is transcribed from the C++ ``consumes<>``/``produces<>`` calls of the
producers in ``RecoHGCal/TICL/plugins`` (see the design notes in the package
README).  ``test/test_catalog_schema.py`` locks this registry against the live
``_cfi`` defaults so that any drift in the raw configuration is detected.
"""

from dataclasses import dataclass, field
from typing import Tuple


# --------------------------------------------------------------------------- #
# Product / consumption descriptors
# --------------------------------------------------------------------------- #

# Instance-label rules for produced products:
#   ""                 -> produced under the module label only (no instance)
#   "fixed:<label>"    -> produced under a fixed instance label
#   "param:<name>"     -> instance label is the value of config parameter <name>
@dataclass(frozen=True)
class Product:
    cpp_type: str
    instance_rule: str = ""

    def instance_label(self, module):
        """Resolve the concrete instance label for a built ``module``."""
        if self.instance_rule.startswith("fixed:"):
            return self.instance_rule[len("fixed:"):]
        if self.instance_rule.startswith("param:"):
            pname = self.instance_rule[len("param:"):]
            return getattr(module, pname).value()
        return ""


@dataclass(frozen=True)
class Consumed:
    param: str          # config parameter holding the InputTag / VInputTag
    cpp_type: str       # required C++ product type
    vector: bool = False  # True for VInputTag (consumes many)


@dataclass(frozen=True)
class ModuleSpec:
    key: str                       # logical class name, e.g. "TrackstersProducer"
    cfi_module: str                # python import path of the default
    cfi_symbol: str                # symbol to import from cfi_module
    produces: Tuple[Product, ...] = ()
    consumes: Tuple[Consumed, ...] = ()
    # InputTag/VInputTag parameters that consume *external* (non-TICL) products
    # (tracks, muons, timing ValueMaps...).  They are not type-checked, but are
    # listed so the drift test can flag any *new, unaccounted* InputTag parameter.
    external_inputs: Tuple[str, ...] = ()
    backends: Tuple[str, ...] = ("cpu",)

    def supports(self, backend):
        return backend in self.backends


# --------------------------------------------------------------------------- #
# C++ product type aliases (kept verbatim so type checks are exact)
# --------------------------------------------------------------------------- #

T_CALOCLUSTERS = "std::vector<reco::CaloCluster>"
T_MASK = "std::vector<float>"
T_TRACKSTERS = "std::vector<ticl::Trackster>"
T_SEEDS = "std::vector<ticl::TICLSeedingRegion>"
T_LINKS = "std::vector<std::vector<unsigned int>>"
T_CANDIDATES = "std::vector<ticl::TICLCandidate>"
T_TIME = "edm::ValueMap<std::pair<float,float>>"
T_TILES = "ticl::TICLLayerTiles"
T_TILES_BARREL = "ticl::TICLLayerTilesBarrel"
T_TILES_HFNOSE = "ticl::TICLLayerTilesHFNose"
T_PFCANDS = "reco::PFCandidateCollection"
T_SUPERCLUSTERS = "reco::SuperClusterCollection"
T_MTDSOA = "MtdHostCollection"


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

_SPECS = [
    ModuleSpec(
        key="TICLLayerTileProducer",
        cfi_module="RecoHGCal.TICL.ticlLayerTileProducer_cfi",
        cfi_symbol="ticlLayerTileProducer",
        produces=(
            Product(T_TILES),
            Product(T_TILES_BARREL, "fixed:ticlLayerTilesBarrel"),
            Product(T_TILES_HFNOSE),
        ),
        consumes=(
            Consumed("layer_clusters", T_CALOCLUSTERS),
            Consumed("layer_HFNose_clusters", T_CALOCLUSTERS),
        ),
    ),
    ModuleSpec(
        key="TICLSeedingRegionProducer",
        cfi_module="RecoHGCal.TICL.ticlSeedingRegionProducer_cfi",
        cfi_symbol="ticlSeedingRegionProducer",
        produces=(Product(T_SEEDS),),
        consumes=(),  # seeding inputs are consumed inside the plugin via cutTk etc.
    ),
    ModuleSpec(
        key="FilteredLayerClustersProducer",
        cfi_module="RecoHGCal.TICL.filteredLayerClustersProducer_cfi",
        cfi_symbol="filteredLayerClustersProducer",
        produces=(Product(T_MASK, "param:iteration_label"),),
        consumes=(
            Consumed("LayerClusters", T_CALOCLUSTERS),
            Consumed("LayerClustersInputMask", T_MASK),
        ),
    ),
    ModuleSpec(
        key="TrackstersProducer",
        cfi_module="RecoHGCal.TICL.trackstersProducer_cfi",
        cfi_symbol="trackstersProducer",
        produces=(
            Product(T_TRACKSTERS),
            Product(T_MASK),
        ),
        consumes=(
            Consumed("layer_clusters", T_CALOCLUSTERS),
            Consumed("filtered_mask", T_MASK),
            Consumed("original_mask", T_MASK),
            Consumed("time_layerclusters", T_TIME),
            Consumed("seeding_regions", T_SEEDS),
            Consumed("layer_clusters_tiles", T_TILES),
            Consumed("layer_clusters_barrel_tiles", T_TILES_BARREL),
            Consumed("layer_clusters_hfnose_tiles", T_TILES_HFNOSE),
        ),
    ),
    ModuleSpec(
        key="TracksterLinksProducer",
        cfi_module="RecoHGCal.TICL.tracksterLinksProducer_cfi",
        cfi_symbol="tracksterLinksProducer",
        produces=(
            Product(T_TRACKSTERS),
            Product(T_LINKS),
            Product(T_LINKS, "fixed:linkedTracksterIdToInputTracksterId"),
            Product(T_MASK),
        ),
        consumes=(
            Consumed("layer_clusters", T_CALOCLUSTERS),
            Consumed("layer_clustersTime", T_TIME),
            Consumed("tracksters_collections", T_TRACKSTERS, vector=True),
            Consumed("original_masks", T_MASK, vector=True),
        ),
    ),
    ModuleSpec(
        key="EGammaSuperclusterProducer",
        cfi_module="RecoHGCal.TICL.ticlEGammaSuperClusterProducer_cfi",
        cfi_symbol="ticlEGammaSuperClusterProducer",
        produces=(
            Product(T_SUPERCLUSTERS),
            Product(T_CALOCLUSTERS),
        ),
        consumes=(
            Consumed("ticlSuperClusters", T_TRACKSTERS),
            Consumed("ticlTrackstersEM", T_TRACKSTERS),
            Consumed("layerClusters", T_CALOCLUSTERS),
        ),
    ),
    ModuleSpec(
        key="TICLCandidateProducer",
        cfi_module="RecoHGCal.TICL.ticlCandidateProducer_cfi",
        cfi_symbol="ticlCandidateProducer",
        produces=(
            Product(T_CANDIDATES),
            Product(T_TRACKSTERS),
        ),
        consumes=(
            Consumed("layer_clusters", T_CALOCLUSTERS),
            Consumed("layer_clustersTime", T_TIME),
            Consumed("egamma_tracksters_collections", T_TRACKSTERS, vector=True),
            Consumed("egamma_tracksterlinks_collections", T_LINKS, vector=True),
            Consumed("general_tracksters_collections", T_TRACKSTERS, vector=True),
            Consumed("general_tracksterlinks_collections", T_LINKS, vector=True),
            Consumed("original_masks", T_MASK, vector=True),
        ),
        external_inputs=("tracks", "muons", "timingSoA"),
    ),
    ModuleSpec(
        key="MTDSoAProducer",
        cfi_module="RecoHGCal.TICL.mtdSoAProducer_cfi",
        cfi_symbol="mtdSoAProducer",
        produces=(Product(T_MTDSOA),),
        consumes=(),  # all inputs are ValueMaps / tracks, not TICL products
        external_inputs=(
            "tracksSrc", "trackAssocSrc", "t0Src", "sigmat0Src", "tmtdSrc",
            "sigmatmtdSrc", "betamtd", "pathmtd", "mvaquality", "posmtd",
            "momentum", "probPi", "probK", "probP",
        ),
    ),
    ModuleSpec(
        key="PFTICLProducer",
        cfi_module="RecoHGCal.TICL.pfTICLProducer_cfi",
        cfi_symbol="pfTICLProducer",
        produces=(Product(T_PFCANDS),),
        consumes=(
            Consumed("ticlCandidateSrc", T_CANDIDATES),
        ),
        external_inputs=("trackTimeValueMap", "trackTimeErrorMap",
                         "trackTimeQualityMap", "muonSrc"),
    ),
]

CATALOG = {spec.key: spec for spec in _SPECS}


# --------------------------------------------------------------------------- #
# Plugin type enumerations (registered factory strings)
# --------------------------------------------------------------------------- #

SEEDING_TYPES = frozenset(
    ["SeedingRegionGlobal", "SeedingRegionByTracks", "SeedingRegionByHF", "SeedingRegionByL1"]
)
FILTER_TYPES = frozenset(
    [
        "ClusterFilterByAlgo",
        "ClusterFilterByAlgoAndSize",
        "ClusterFilterBySize",
        "ClusterFilterByAlgoAndSizeAndLayerRange",
    ]
)
PATTERN_TYPES = frozenset(["CLUE3D", "CA", "FastJet", "Recovery"])
LINKING_TYPES = frozenset(
    ["Skeletons", "SuperClusteringDNN", "SuperClusteringMustache", "FastJet", "Recovery"]
)
INFERENCE_TYPES = frozenset(
    ["TracksterInferenceByCNN", "TracksterInferenceByDNN", "TracksterInferenceByPFN"]
)

# Canonical shared seeding-region module label per seeding type (HGCAL endcap).
SEEDING_MODULE_LABEL = {
    "SeedingRegionGlobal": "ticlSeedingGlobal",
    "SeedingRegionByTracks": "ticlSeedingTrk",
}

# Backends
CPU = "cpu"
GPU = "gpu"
