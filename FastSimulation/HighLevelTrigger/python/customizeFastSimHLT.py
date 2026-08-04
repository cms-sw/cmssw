import os
import re
import time

import FWCore.ParameterSet.Config as cms

from FastSimulation.HighLevelTrigger.full2fast import (
    modify_hltL3TrajSeedIOHit,
    modify_hltL3TrajSeedOIHit,
)

_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")

_DEFAULT_MUON_L1_SEED_MODULES = (
    # Low-threshold single-muon paths feed HLTMuonL1TFilter from these
    # real object-map seeds; the broad L1GlobalDecision fallback carries
    # no usable muon refs for the following L1T filters.
    "hltL1sSingleMu3IorSingleMu5IorSingleMu7",
    "hltL1sSingleMu7",
    "hltL1sSingleMu7DQ",
    "hltL1sSingleMu15DQ",
    "hltL1sSingleMu15DQorSingleMu7",
    "hltL1sSingleMu18",
    "hltL1sSingleMu22or25",
    "hltL1sSingleMu22",
    # Low-pT muon cross paths feed HLTMuonL1TFilter from this object-map seed.
    "hltL1sSingleMuOpenObjectMap",
)

_DEFAULT_CROSS_OBJECT_PF_L1_SEED_MODULES = (
    # Double-muon+MET/PFMHT paths need this real seed's object map; the broad
    # L1GlobalDecision fallback carries no muon refs to the next L1T filter.
    "hltL1sDoubleMu0ETM40IorDoubleMu0ETM55IorDoubleMu0ETM60IorDoubleMu0ETM65IorDoubleMu0ETM70",
)

_DEFAULT_JET_MET_BTAG_L1_SEED_MODULES = (
    # Audited against the active auto:phase1_2023_realistic Run-3 L1 menu and
    # restricted to candidate jet/MET/b-tag/tau paths from the FastSim path audit.
    "hltL1Mu3er1p5Jet100er2p5ETMHF40ORETMHF50",
    "hltL1sAllETMHFHTT60Seeds",
    "hltL1sAllHTTSeeds",
    "hltL1sDiJet120er2p5Mu3dRMax0p8",
    "hltL1sDiJet16er2p5Mu3dRMax0p4",
    "hltL1sDiJet35er2p5Mu3dRMax0p4",
    "hltL1sDiJet60er2p5Mu3dRMax0p4",
    "hltL1sDiJet80er2p5Mu3dRMax0p4",
    "hltL1sDoubleMu0ETM40IorDoubleMu0ETM55IorDoubleMu0ETM60IorDoubleMu0ETM65IorDoubleMu0ETM70",
    "hltL1sDoubleMu0Jet90er2p5dRMax0p8dRMu1p6",
    "hltL1sDoubleMu125to157",
    "hltL1sDoublePFJet40er2p5",
    "hltL1sEG40To45IorJet170To200IorHTT300To500IorETM70ToETM150",
    "hltL1sHTT120er",
    "hltL1sHTT160er",
    "hltL1sHTT200er",
    "hltL1sHTT255er",
    "hltL1sHTT280to500erIorHTT250to340erQuadJet",
    "hltL1sHTT280to500erIorHTT250to340erQuadJetTripleJet",
    "hltL1sMu3Jet30er2p5",
    "hltL1sQuadJetC50to60IorHTT280to500IorHTT250to340QuadJet",
    "hltL1sQuadJetOrHTTOrMuonHTT",
    "hltL1sSingleJet120",
    "hltL1sSingleJet120Fwd",
    "hltL1sSingleJet60Fwd",
    "hltL1sSingleJet90",
    "hltL1sSingleJet90Fwd",
    "hltL1sSingleMu3IorMu3Jet30er2p5",
    "hltL1sTripleJet1008572VBFIorHTTIorDoubleJetCIorSingleJet",
    "hltL1sTripleJet1058576VBFIorHTTIorDoubleJetCIorSingleJet",
    "hltL1sTripleJetVBFIorHTTIorSingleJet",
    "hltL1sVoMu6HTT240Or250",
    "hltL1sZeroBias",
)

_DEFAULT_OBJECT_L1_SEED_MODULES = (
    # Calo jet/MET/HT paths that already run without PF tracking dependencies.
    "hltL1sAllETMHFSeeds",
    "hltL1sHTT280orHTT320orHTT360orETT2000",
    "hltL1sSingleJet170IorSingleJet180IorSingleJet200",
    "hltL1sVoHTT200SingleLLPJet60",

    # Simple b-tag/muon-jet seed.
    "hltL1sSingleJet200",
)

_DEFAULT_EGAMMA_TAU_L1_SEED_MODULES = (
    # Audited against the active auto:phase1_2023_realistic Run-3 L1 menu.
    # Deliberately excludes seeds whose expressions contain missing L1 algos:
    # hltL1VBFLooseIsoEG, hltL1sDoubleEGXer1p2dRMaxY,
    # hltL1sSingleEG20er2p5, and hltL1sTauVeryBigOR.
    "hltL1sBigORDoubleLooseIsoEGXXer",
    "hltL1sBigORLooseIsoEGXXerIsoTauYYerdRMin0p3",
    "hltL1sBigORLooseIsoEGXXerIsoTauYYerdRMin0p3ORSingleEG36er",
    "hltL1sBigORMu18erTauXXer2p1",
    "hltL1sBigOrMuXXerIsoTauYYer",
    "hltL1sDoubleEG6to8HTT250to300IorL1sHTT",
    "hltL1sDoubleMu0er2p0SQOSdEtaMax1p6orTripleMu21p50",
    "hltL1sDoubleMu157IorDoubleMu4p5SQOSdR1p2IorSingleMu25IorSingleMu22erIorTripleMuMassMax9",
    "hltL1sDoubleMu3DoubleEG7p5",
    "hltL1sDoubleMu4EG9",
    "hltL1sDoubleMu5DoubleEG3",
    "hltL1sDoubleMu7EG7",
    "hltL1sDoubleMuForBsToMMG",
    "hltL1sDoubleTauBigOR",
    "hltL1sHTT200erFromObjectMap",
    "hltL1sHTT380erIorHTT320er",
    "hltL1sMu23EG10IorMu20EG17",
    "hltL1sMu5EG23IorMu5IsoEG20IorMu7EG23IorMu7IsoEG20IorMuIso7EG23",
    "hltL1sMu5EG23IorMu7EG23IorMu20EG17IorMu23EG10",
    "hltL1sMu18erTau26er2p1Jet55",
    "hltL1sMu18erTau26er2p1Jet70",
    "hltL1sMu6DoubleEG10",
    "hltL1sMu6HTT240",
    "hltL1sMu22erIsoTau40er",
    "hltL1sMuShowerOneNominal",
    "hltL1sSingleAndDoubleEG",
    "hltL1sSingleAndDoubleEGNonIsoForDisplacedTrig",
    "hltL1sSingleAndDoubleEGNonIsoOrWithEG26WithJetAndTau",
    "hltL1sSingleAndDoubleEGNonIsoOrWithJetAndTau",
    "hltL1sSingleAndDoubleEGor",
    "hltL1sSingleEG10IorSingleEG15",
    "hltL1sSingleEG10IorSingleEG5",
    "hltL1sSingleEG10er2p5",
    "hltL1sSingleEG10er2p5SingleEG15er2p5",
    "hltL1sSingleEG15er2p5",
    "hltL1sSingleEG26",
    "hltL1sSingleEG34to45",
    "hltL1sSingleEG34to50",
    "hltL1sSingleEG40to50",
    "hltL1sSingleEG5ObjectMap",
    "hltL1sSingleEGNonIsoOrWithJetAndTau",
    "hltL1sSingleEGNonIsoOrWithJetAndTauNoPS",
    "hltL1sSingleEGor",
    "hltL1sSingleIsoEG28to45",
    "hltL1sSingleMu22or25",
    "hltL1sSingleMu25",
    "hltL1sSingleTau",
    "hltL1sTripleMuControl",
    "hltL1sVeryBigORMu18erTauXXer2p1",
)

_UNSUPPORTED_EGAMMA_TAU_L1_SEED_MODULES = (
    "hltL1VBFLooseIsoEG",
    "hltL1sDoubleEGXer1p2dRMaxY",
    "hltL1sSingleEG20er2p5",
    "hltL1sTauVeryBigOR",
)

_UNSUPPORTED_JET_MET_BTAG_TAU_L1_SEED_MODULES = (
    "hltL1sDoublePFJet20er2p5",
    "hltL1sDoublePFJet30er2p5",
    "hltL1sHTT100er",
    "hltL1sQuadPFJet20er2p5",
    "hltL1sQuadPFJet24er2p5",
    "hltL1sSingleJet24BptxAND",
    "hltL1sSingleJet35PlusBptxAND",
    "hltL1sSingleJet60PlusBptxAND",
    "hltL1sSingleJet90withBptxAND",
    "hltL1sTauVeryBigOR",
    "hltL1sTripleJet957565VBFIorHTTIorDoubleJetCIorSingleJetorQuadJet95756520",
    "hltL1sTriplePFJet20er2p5",
    "hltL1sTriplePFJet30er2p5",
)

_DEFAULT_PNET_BTAG_TAU_L1_SEED_MODULES = (
    "hltL1sDoubleTauBigOR",
    "hltL1sSingleTau",
)


def _modify_if_present(process, module_name, modifier):
    if not hasattr(process, module_name):
        return

    try:
        modifier(getattr(process, module_name))
    except AttributeError as exc:
        print("FastSim HLT: skipped %s (%s)" % (module_name, exc))


def _configured_keep_paths():
    raw_paths = os.environ.get("FASTSIM_HLT_KEEP_PATHS", "")
    return set(path.strip() for path in raw_paths.split(",") if path.strip())


def _configured_only_paths():
    raw_paths = os.environ.get("FASTSIM_HLT_ONLY_PATHS", "")
    return set(path.strip() for path in raw_paths.split(",") if path.strip())


def _configured_keep_l1_seed_modules():
    raw_modules = os.environ.get("FASTSIM_HLT_KEEP_L1_SEED_MODULES", "")
    return set(module.strip() for module in raw_modules.split(",") if module.strip())


def _configured_keep_l1_seed_families():
    raw_families = os.environ.get("FASTSIM_HLT_KEEP_L1_SEED_FAMILIES", "")
    return set(family.strip() for family in raw_families.split(",") if family.strip())


def _configured_bool(name, default=False):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    value = raw_value.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False

    print("FastSim HLT: unrecognized value for %s=%r; using default %s" %
          (name, raw_value, default))
    return default


def _configured_enable_review_default():
    return _configured_bool("FASTSIM_HLT_ENABLE_REVIEW_DEFAULT")


def _configured_enable_muon_prototype(default=False):
    return _configured_bool("FASTSIM_HLT_ENABLE_MUON_PROTOTYPE", default)


def _configured_enable_muon_pixel_tracks_prototype(default=False):
    return _configured_bool("FASTSIM_HLT_ENABLE_MUON_PIXEL_TRACKS_PROTOTYPE", default)


def _configured_enable_muon_iso_prototype(default=False):
    return _configured_bool("FASTSIM_HLT_ENABLE_MUON_ISO_PROTOTYPE", default)


def _configured_enable_object_prototypes(default=False):
    return _configured_bool("FASTSIM_HLT_ENABLE_OBJECT_PROTOTYPES", default)


def _configured_enable_jet_met_btag_l1_prototypes(default=False):
    return _configured_bool("FASTSIM_HLT_ENABLE_JET_MET_BTAG_L1_PROTOTYPES", default)


def _configured_enable_egamma_tau_l1_prototypes(default=False):
    return _configured_bool("FASTSIM_HLT_ENABLE_EGAMMA_TAU_L1_PROTOTYPES", default)


def _configured_enable_egamma_gsf_prototype(default=False):
    return _configured_bool("FASTSIM_HLT_ENABLE_EGAMMA_GSF_PROTOTYPE", default)


def _configured_enable_egamma_mkfit_iso_prototype(default=False):
    return _configured_bool("FASTSIM_HLT_ENABLE_EGAMMA_MKFIT_ISO_PROTOTYPE", default)


def _configured_enable_pfmet_mkfit_prototype(default=False):
    return _configured_bool("FASTSIM_HLT_ENABLE_PFMET_MKFIT_PROTOTYPE", default)


def _configured_enable_high_pfmet_prototype(default=False):
    return _configured_bool("FASTSIM_HLT_ENABLE_HIGH_PFMET_PROTOTYPE", default)


def _configured_enable_pfjet_pfht_mkfit_prototype(default=False):
    return _configured_bool("FASTSIM_HLT_ENABLE_PFJET_PFHT_MKFIT_PROTOTYPE", default)


def _configured_enable_ak8_softdrop_prototype(default=False):
    return _configured_bool("FASTSIM_HLT_ENABLE_AK8_SOFTDROP_PROTOTYPE", default)


def _configured_enable_cross_object_pf_mkfit_prototype(default=False):
    return _configured_bool("FASTSIM_HLT_ENABLE_CROSS_OBJECT_PF_MKFIT_PROTOTYPE", default)


def _configured_enable_pnet_vertex_prototype(default=False):
    return _configured_bool("FASTSIM_HLT_ENABLE_PNET_VERTEX_PROTOTYPE", default)


def _configured_enable_pnet_btag_vertex_prototype(default=False):
    return _configured_bool("FASTSIM_HLT_ENABLE_PNET_BTAG_VERTEX_PROTOTYPE", default)


def _configured_pnet_vertex_source():
    return os.environ.get("FASTSIM_HLT_PNET_VERTEX_SOURCE", "hltVerticesPF:WithBS").strip() or "hltVerticesPF:WithBS"


def _configured_pnet_vertex_cut():
    return os.environ.get("FASTSIM_HLT_PNET_VERTEX_CUT", "!isFake").strip() or "!isFake"


def _configured_pnet_vertex_track_source():
    return os.environ.get(
        "FASTSIM_HLT_PNET_VERTEX_TRACK_SOURCE",
        "generalTracksBeforeMixing",
    ).strip()


def _configured_pnet_btag_vertex_source():
    return os.environ.get(
        "FASTSIM_HLT_PNET_BTAG_VERTEX_SOURCE",
        "offlineSlimmedPrimaryVertices",
    ).strip() or "offlineSlimmedPrimaryVertices"


def _configured_pnet_btag_vertex_cut():
    return os.environ.get(
        "FASTSIM_HLT_PNET_BTAG_VERTEX_CUT",
        "!isFake && ndof >= 4",
    ).strip() or "!isFake && ndof >= 4"


def _configured_pnet_btag_vertex_include_tau_cross_paths():
    return _configured_bool("FASTSIM_HLT_PNET_BTAG_VERTEX_INCLUDE_TAU_CROSS_PATHS")


def _configured_relax_l3_muon_id():
    return _configured_bool("FASTSIM_HLT_RELAX_L3_MUON_ID")


def _configured_use_fast_l3_muon_seeds(default=False):
    return _configured_bool("FASTSIM_HLT_USE_FAST_L3_MUON_SEEDS", default)


def _configured_use_fast_l3_muon_track_candidates(default=False):
    return _configured_bool("FASTSIM_HLT_USE_FAST_L3_MUON_TRACK_CANDIDATES", default)


def _configured_muon_iso_track_source():
    return os.environ.get("FASTSIM_HLT_MUON_ISO_TRACK_SOURCE", "").strip()


def _configured_muon_iso_track_quality():
    return os.environ.get("FASTSIM_HLT_MUON_ISO_TRACK_QUALITY", "").strip()


def _configured_keep_debug_products():
    return _configured_bool("FASTSIM_HLT_KEEP_DEBUG_PRODUCTS")


def _configured_object_l1_save_tags():
    return _configured_bool("FASTSIM_HLT_OBJECT_L1_SAVE_TAGS")


def _configured_profile_customizer():
    return _configured_bool("FASTSIM_HLT_PROFILE_CUSTOMIZER")


def _configured_use_combined_pruner(default=False):
    return _configured_bool("FASTSIM_HLT_USE_COMBINED_PRUNER", default)


class _CustomizerProfiler(object):
    def __init__(self, enabled):
        self._enabled = enabled
        self._start = time.monotonic()
        self._previous = self._start

    def mark(self, label):
        if not self._enabled:
            return

        now = time.monotonic()
        print(
            "FastSim HLT profile: %-42s step=%8.3fs total=%8.3fs" %
            (label, now - self._previous, now - self._start)
        )
        self._previous = now


def _base_path_name(name):
    stem, version, suffix = name.rpartition("_v")
    if version and suffix.isdigit():
        return stem

    return name


def _path_requested(name, requested_paths):
    return name in requested_paths or _base_path_name(name) in requested_paths


def _scheduled_path_names(process):
    if not hasattr(process, "schedule"):
        return set()

    scheduled_ids = set(id(path) for path in process.schedule)
    return set(
        path_name
        for path_name, path in process.paths_().items()
        if id(path) in scheduled_ids
    )


def _is_hlt_menu_path(name):
    return (
        name.startswith("HLT_") or
        name.startswith("Dataset_") or
        name.startswith("AlCa_") or
        name.startswith("DST_") or
        name.startswith("MC_") or
        name == "HLTriggerFinalPath"
    )


def _path_family(name):
    if "Scouting" in name:
        return "scouting"
    if name.startswith("Dataset_"):
        return "dataset"
    if name.startswith("MC_"):
        return "mc"
    if name.startswith("AlCa_"):
        return "alca"
    if name.startswith("DST_"):
        return "dst"
    if re.search(r"(BTag|BTagMu|DeepJet|BJet)", name):
        return "btag"
    if re.search(r"(Ele|Photon|Egamma|EG)", name):
        return "egamma"
    if re.search(r"(PFHT|HT|PFMET|MET|PFJet|AK4|AK8|CaloJet|QuadPFJet|DiPFJet)", name):
        return "jet_met"
    if "Tau" in name:
        return "tau"
    if re.search(r"(Mu|TkMu|L2Mu|L3Mu)", name):
        return "muon"
    return "other"


def _pfmet_mkfit_prototype_paths(process):
    selected = []
    for name in process.paths_():
        base = _base_path_name(name)
        if "PFMET" in base and "PFMHT" in base and "IDTight" in base:
            selected.append(base)

    return sorted(set(selected))


_PFJET_PFHT_PROTOTYPE_EXCLUDE_RE = re.compile(
    r"(PNet|BTag|DeepJet|BJet|Scouting|SoftDrop|Mass|PPS|Displaced|Delayed|"
    r"CICADA|AXO|Topo|CscCluster|Tau|Ele|Photon|Mu|MET|MHT|NoMu|Nch)"
)


def _is_simple_pfjet_pfht_path(base):
    if not base.startswith("HLT_"):
        return False
    if _PFJET_PFHT_PROTOTYPE_EXCLUDE_RE.search(base):
        return False
    if re.fullmatch(r"HLT_(PFJet|PFJetFwd|AK8PFJet|AK8PFJetFwd)\d+", base):
        return True
    if re.fullmatch(r"HLT_PFJet\d+_L1Jet\d+(withBptxAND)?", base):
        return True
    if re.fullmatch(r"HLT_PFHT\d+", base):
        return True
    if re.fullmatch(r"HLT_PFHT\d+_QuadPFJet\d+", base):
        return True
    if re.fullmatch(r"HLT_PFHT\d+PT\d+_QuadPFJet_\d+_\d+_\d+_\d+", base):
        return True
    if re.fullmatch(r"HLT_PFHT\d+_(FivePFJet|SixPFJet)_\d+(?:_\d+)*", base):
        return True
    if re.fullmatch(r"HLT_PFHT\d+_(FivePFJet|SixPFJet)\d+(?:_\d+)*", base):
        return True
    if re.fullmatch(r"HLT_(DiPFJetAve|DoublePFJet|TriplePFJet|QuadPFJet)\d+", base):
        return True
    if re.fullmatch(r"HLT_QuadPFJet\d+_\d+_\d+_\d+", base):
        return True
    if re.fullmatch(r"HLT_QuadPFJet\d+_L1QuadJet\d+", base):
        return True

    return False


def _pfjet_pfht_mkfit_prototype_paths(process):
    selected = []
    for name in process.paths_():
        base = _base_path_name(name)
        if _is_simple_pfjet_pfht_path(base):
            selected.append(base)

    return sorted(set(selected))


def _is_high_pfmet_path(base):
    if not base.startswith("HLT_"):
        return False
    if re.search(r"(NoMu|DTCluster|CscCluster|Displaced|Delayed|Scouting)", base):
        return False
    if re.fullmatch(r"HLT_PFMET\d+_(?:NotCleaned|BeamHaloCleaned)", base):
        return True
    if re.fullmatch(r"HLT_PFMETTypeOne\d+_BeamHaloCleaned", base):
        return True

    return False


def _high_pfmet_prototype_paths(process):
    selected = []
    for name in process.paths_():
        base = _base_path_name(name)
        if _is_high_pfmet_path(base):
            selected.append(base)

    return sorted(set(selected))


def _is_ak8_softdrop_path(base):
    if not base.startswith("HLT_"):
        return False
    if re.search(r"(PNet|ParticleNet|GloParT|BTag|AK15|Scouting|Displaced|Delayed|CscCluster|DTCluster)", base):
        return False
    if re.fullmatch(r"HLT_AK8PFJet\d+_SoftDropMass\d+", base):
        return True
    if re.fullmatch(r"HLT_AK8DiPFJet\d+_\d+_SoftDropMass\d+", base):
        return True

    return False


def _ak8_softdrop_prototype_paths(process):
    selected = []
    for name in process.paths_():
        base = _base_path_name(name)
        if _is_ak8_softdrop_path(base):
            selected.append(base)

    return sorted(set(selected))


_CROSS_OBJECT_PF_PROTOTYPE_EXCLUDE_RE = re.compile(
    r"(PNet|BTag|DeepJet|BJet|Scouting|SoftDrop|Mass|PPS|Displaced|Delayed|"
    r"CICADA|AXO|Topo|VBF|Mjj|MJJ|CscCluster|DTCluster|Tau|NoMu|GPUvsCPU|"
    r"Beamspot|LLP|DoubleAK4|QuadPFJet)"
)


def _is_cross_object_pf_path(base):
    if not base.startswith("HLT_"):
        return False
    if _CROSS_OBJECT_PF_PROTOTYPE_EXCLUDE_RE.search(base):
        return False

    pf_object = r"(?:AK8PFJet|PFJet|PFHT|PFMET|PFMHT)\d+"
    if re.fullmatch(r"HLT_DoubleMu3_(?:DCA|DZ)_PFMET\d+_PFMHT\d+", base):
        return True
    if re.fullmatch(r"HLT_Ele\d+.*_(?:%s)(?:_PFMET\d+)?" % pf_object, base):
        return True
    if re.fullmatch(r"HLT_(?:IsoMu|Mu)\d+(?:eta[\w]+)?.*_(?:%s)(?:_PFMET\d+)?" % pf_object, base):
        return True
    if re.fullmatch(r"HLT_Mu\d+_.*_Ele\d+_.*_PFHT\d+", base):
        return True
    if re.fullmatch(r"HLT_Photon\d+EB_TightID_TightIso_(?:AK8PFJet|PFJet)\d+", base):
        return True

    return False


def _cross_object_pf_mkfit_prototype_paths(process):
    selected = []
    for name in process.paths_():
        base = _base_path_name(name)
        if _is_cross_object_pf_path(base):
            selected.append(base)

    return sorted(set(selected))


def _is_egamma_mkfit_iso_path(base):
    if not base.startswith("HLT_"):
        return False
    if re.fullmatch(r"HLT_Ele(?:30|32|35|38|40)_WPTight_Gsf", base):
        return True
    if base == "HLT_Ele32_WPTight_Gsf_L1DoubleEG":
        return True
    if re.fullmatch(r"HLT_Photon(?:30|40|50)EB_TightID_TightIso", base):
        return True

    return False


def _egamma_mkfit_iso_prototype_paths(process):
    selected = []
    for name in process.paths_():
        base = _base_path_name(name)
        if _is_egamma_mkfit_iso_path(base):
            selected.append(base)

    return sorted(set(selected))


_PNET_BTAG_TAU_PROTOTYPE_EXCLUDE_RE = re.compile(
    r"(SoftDrop|PNetBB|PNetV02BB|PNetTauTau|GloParT|AK15|Nch|Topo|"
    r"Scouting|CscCluster|DTCluster|Displaced|Delayed)"
)


def _is_pnet_btag_tau_path(base):
    if not base.startswith("HLT_"):
        return False
    if _PNET_BTAG_TAU_PROTOTYPE_EXCLUDE_RE.search(base):
        return False
    return bool(re.search(r"PNet(?:\d+)?(?:BTag|Tauh)", base))


def _pnet_btag_tau_prototype_paths(process):
    selected = []
    for name in process.paths_():
        base = _base_path_name(name)
        if _is_pnet_btag_tau_path(base):
            selected.append(base)

    return sorted(set(selected))


_REVIEW_DEFAULT_UNSUPPORTED_PATH_RE = re.compile(
    r"(PNet|ParticleNet|BTag|BJet|DeepJet|Tau|Scouting|CICADA|AXO|"
    r"DTCluster|CscCluster|Displaced|Delayed|PPS|GloParT|AK15|GPUvsCPU|"
    r"L1SingleMuShower|L1SingleMuCosmics|Cosmic|NoVtx)"
)


def _remove_review_default_unsupported_paths(process, keep_paths=()):
    """Prune known non-review families unless the caller explicitly keeps them."""
    removed = []
    if not hasattr(process, "schedule"):
        return removed

    for name, path in list(process.paths_().items()):
        base = _base_path_name(name)
        if _path_requested(name, keep_paths):
            continue
        if not _is_hlt_menu_path(name):
            continue
        if not _REVIEW_DEFAULT_UNSUPPORTED_PATH_RE.search(base):
            continue

        try:
            process.schedule.remove(path)
            removed.append(name)
        except ValueError:
            pass

    return removed


def _remove_unrequested_hlt_menu_paths(process, only_paths):
    if not hasattr(process, "schedule"):
        return []

    removed = []
    for collection in (process.paths_(), process.endpaths_()):
        for name, path in list(collection.items()):
            if not _is_hlt_menu_path(name) or _path_requested(name, only_paths):
                continue

            try:
                process.schedule.remove(path)
                removed.append(name)
            except ValueError:
                pass

    return removed


def _remove_scheduled_paths(process, prefix, keep_paths=()):
    if not hasattr(process, "schedule"):
        return []

    removed = []
    for name, path in list(process.paths_().items()):
        if _path_requested(name, keep_paths):
            continue

        if not name.startswith(prefix):
            continue

        try:
            process.schedule.remove(path)
            removed.append(name)
        except ValueError:
            pass

    return removed


def _remove_scheduled_paths_using(process, needle, keep_paths=()):
    if not hasattr(process, "schedule"):
        return []

    removed = []
    for name, path in list(process.paths_().items()):
        if _path_requested(name, keep_paths):
            continue

        try:
            path_definition = path.dumpPython()
        except Exception:
            continue

        if needle not in path_definition:
            continue

        try:
            process.schedule.remove(path)
            removed.append(name)
        except ValueError:
            pass

    return removed


def _remove_scheduled_paths_with_modules(process, module_names, keep_paths=()):
    if not hasattr(process, "schedule"):
        return []

    unsupported = set(module_names)
    removed = []
    for name, path in list(process.paths_().items()):
        if _path_requested(name, keep_paths):
            continue

        try:
            path_modules = set(path.moduleNames())
        except Exception:
            continue

        if path_modules.isdisjoint(unsupported):
            continue

        try:
            process.schedule.remove(path)
            removed.append(name)
        except ValueError:
            pass

    return removed


def _remove_scheduled_paths_with_module_name_fragment(process, fragment, keep_paths=()):
    if not hasattr(process, "schedule"):
        return []

    removed = []
    for name, path in list(process.paths_().items()):
        if _path_requested(name, keep_paths):
            continue

        try:
            path_modules = path.moduleNames()
        except Exception:
            continue

        if not any(fragment in module_name for module_name in path_modules):
            continue

        try:
            process.schedule.remove(path)
            removed.append(name)
        except ValueError:
            pass

    return removed


def _remove_scheduled_paths_with_combined_pruner(process, keep_paths=()):
    empty = {
        "mc": [],
        "mkfit_string": [],
        "mkfit_module": [],
        "mkfit_any": [],
        "tracking": [],
    }
    if not hasattr(process, "schedule"):
        return empty

    mkfit_seed_modules = set((
        "hltDoubletRecoveryPFlowCkfTrackCandidatesMkFitSeeds",
        "hltIter0PFlowCkfTrackCandidatesMkFitSeeds",
    ))
    tracking_sequence_needles = (
        "HLTTrackingForBeamSpot",
        "HLTPFScoutingTrackingSequence",
    )
    removed = dict((key, []) for key in empty)

    for name, path in list(process.paths_().items()):
        if _path_requested(name, keep_paths):
            continue

        category = None
        path_definition = None

        def dumped_path():
            nonlocal path_definition
            if path_definition is not None:
                return path_definition
            try:
                path_definition = path.dumpPython()
            except Exception:
                path_definition = ""
            return path_definition

        if name.startswith("MC_"):
            category = "mc"
        elif "hltIter0PFlowCkfTrackCandidatesMkFitSeeds" in dumped_path():
            category = "mkfit_string"
        else:
            try:
                path_modules = path.moduleNames()
            except Exception:
                path_modules = ()

            if not set(path_modules).isdisjoint(mkfit_seed_modules):
                category = "mkfit_module"
            elif any("MkFit" in module_name for module_name in path_modules):
                category = "mkfit_any"
            elif any(needle in dumped_path() for needle in tracking_sequence_needles):
                category = "tracking"

        if category is None:
            continue

        try:
            process.schedule.remove(path)
            removed[category].append(name)
        except ValueError:
            pass

    return removed


def _format_path_list(paths, limit=20):
    if len(paths) <= limit:
        return ", ".join(paths)

    return "%s, ... (%d total)" % (", ".join(paths[:limit]), len(paths))


def _ensure_audit(process):
    if not hasattr(process, "fastSimHLTAudit"):
        process.fastSimHLTAudit = cms.PSet()

    return process.fastSimHLTAudit


def _set_audit_vstring(process, name, values):
    setattr(_ensure_audit(process), name, cms.vstring(*[str(value) for value in values]))


def _set_audit_bool(process, name, value):
    setattr(_ensure_audit(process), name, cms.bool(bool(value)))


def _input_tag_from_text(text, fallback="hltVerticesPF"):
    parts = [part.strip() for part in str(text).split(":")]
    if len(parts) == 1 and parts[0]:
        return cms.InputTag(parts[0])
    if len(parts) == 2 and parts[0]:
        return cms.InputTag(parts[0], parts[1])
    if len(parts) == 3 and parts[0]:
        return cms.InputTag(parts[0], parts[1], parts[2])

    print("FastSim HLT: unsupported InputTag text '%s'; using %s" % (text, fallback))
    return cms.InputTag(fallback)


def _input_tag_parts(tag):
    if hasattr(tag, "getModuleLabel"):
        return [
            tag.getModuleLabel(),
            tag.getProductInstanceLabel(),
            tag.getProcessName(),
        ]

    parts = [part.strip() for part in str(tag).split(":")]
    while len(parts) < 3:
        parts.append("")
    return parts[:3]


def _retarget_input_tag(tag, old_module, new_module):
    parts = _input_tag_parts(tag)
    if parts[0] != old_module:
        return tag

    parts[0] = new_module
    if parts[2]:
        return cms.InputTag(parts[0], parts[1], parts[2])
    if parts[1]:
        return cms.InputTag(parts[0], parts[1])
    return cms.InputTag(parts[0])


def _retarget_vinput_tag(tags, old_module, new_module):
    return cms.VInputTag(*[
        _retarget_input_tag(tag, old_module, new_module)
        for tag in tags
    ])


def _add_direct_mkfit_geometry_esproducer(process):
    process.add_(cms.ESProducer("MkFitGeometryESProducer"))


def _reroute_pfmet_mht_to_corrected_pfjets(process):
    changed = []
    for module_name, module in process.producers_().items():
        if module.type_() != "HLTHtMhtProducer":
            continue
        if not hasattr(module, "jetsLabel"):
            continue
        if "hltAK4PFJetsTightIDCorrected" not in str(module.jetsLabel):
            continue
        module.jetsLabel = cms.InputTag("hltAK4PFJetsCorrected")
        changed.append(module_name)

    return sorted(changed)


def _reroute_pfjet_id_filters_to_corrected_pfjets(process):
    changed = []
    fastsim_hostile_inputs = (
        "hltAK4PFJetsLooseIDCorrected",
        "hltAK4PFJetsTightIDCorrected",
    )
    for module_name, module in process.filters_().items():
        if module.type_() != "HLT1PFJet":
            continue
        if not hasattr(module, "inputTag"):
            continue
        if not any(label in str(module.inputTag) for label in fastsim_hostile_inputs):
            continue
        module.inputTag = cms.InputTag("hltAK4PFJetsCorrected")
        changed.append(module_name)

    return sorted(changed)


def _reroute_muon_iso_track_sources(process, track_source):
    if not track_source:
        return []

    source_tag = _input_tag_from_text(track_source, fallback="hltMergedTracks")
    changed = []
    for module_name, module in process.producers_().items():
        if module.type_() != "L3MuonCombinedRelativeIsolationProducer":
            continue
        if not hasattr(module, "TrkExtractorPSet"):
            continue
        if not hasattr(module.TrkExtractorPSet, "inputTrackCollection"):
            continue

        old_source = str(module.TrkExtractorPSet.inputTrackCollection)
        module.TrkExtractorPSet.inputTrackCollection = source_tag
        changed.append("%s:%s->%s" % (module_name, old_source, track_source))

    return sorted(changed)


def _set_muon_iso_track_selection_quality(process, quality):
    if not quality:
        return []

    changed = []
    for module_name, module in process.producers_().items():
        if module.type_() != "TrackCollectionFilterCloner":
            continue
        if "L3Muon" not in module_name or "TrackSelectionHighPurity" not in module_name:
            continue
        if not hasattr(module, "minQuality"):
            continue

        old_quality = str(module.minQuality)
        module.minQuality = cms.string(quality)
        changed.append("%s:%s->%s" % (module_name, old_quality, quality))

    return sorted(changed)


def _relax_pnet_vertex_filters_for_fastsim(process, vertex_source, vertex_cut, vertex_track_source):
    changed = []
    if vertex_track_source and hasattr(process, "hltVerticesPF"):
        producer = process.hltVerticesPF
        if hasattr(producer, "TrackLabel"):
            producer.TrackLabel = _input_tag_from_text(vertex_track_source)
            changed.append("hltVerticesPF:TrackLabel")

    if hasattr(process, "hltVerticesPFSelector"):
        selector = process.hltVerticesPFSelector
        if hasattr(selector, "filterParams") and hasattr(selector.filterParams, "minNdof"):
            selector.filterParams.minNdof = cms.double(0.0)
            changed.append("hltVerticesPFSelector:minNdof")

    if hasattr(process, "hltVerticesPFFilter"):
        vertex_filter = process.hltVerticesPFFilter
        if hasattr(vertex_filter, "src"):
            # The selector often rejects FastSim's fallback vertices.  For the
            # diagnostic PNet prototype, feed a configured existing vertex
            # collection directly to the HLT vertex filter and let the cut below
            # decide whether to keep it.
            vertex_filter.src = _input_tag_from_text(vertex_source)
            changed.append("hltVerticesPFFilter:src")
        if hasattr(vertex_filter, "cut"):
            vertex_filter.cut = cms.string(vertex_cut)
            changed.append("hltVerticesPFFilter:cut")

    return sorted(set(changed))


def _is_pnet_btag_vertex_path(name, include_tau_cross_paths=False):
    base = _base_path_name(name)
    if not re.search(r"PNet(?:\d+)?BTag", base):
        return False
    if "Tauh" in base and not include_tau_cross_paths:
        return False
    return True


def _install_pnet_btag_vertex_filter_for_fastsim(
    process,
    vertex_source,
    vertex_cut,
    include_tau_cross_paths=False,
):
    def clone_module(old_name, new_name, **updates):
        if not hasattr(process, old_name):
            return None
        clone = getattr(process, old_name).clone()
        for parameter_name, value in updates.items():
            setattr(clone, parameter_name, value)
        setattr(process, new_name, clone)
        return new_name

    cloned_modules = []
    vertex_filter_name = clone_module(
        "hltVerticesPFFilter",
        "hltVerticesPFFilterForPNetBTagFastSim",
        src=_input_tag_from_text(vertex_source, fallback="offlineSlimmedPrimaryVertices"),
        cut=cms.string(vertex_cut),
    )
    if vertex_filter_name is None:
        return [], []
    cloned_modules.append(vertex_filter_name)

    chain_clones = (
        clone_module(
            "hltDeepBLifetimeTagInfosPF",
            "hltDeepBLifetimeTagInfosPFForPNetBTagFastSim",
            primaryVertex=cms.InputTag(vertex_filter_name),
        ),
        clone_module(
            "hltDeepInclusiveVertexFinderPF",
            "hltDeepInclusiveVertexFinderPFForPNetBTagFastSim",
            primaryVertices=cms.InputTag(vertex_filter_name),
        ),
        clone_module(
            "hltDeepInclusiveSecondaryVerticesPF",
            "hltDeepInclusiveSecondaryVerticesPFForPNetBTagFastSim",
            secondaryVertices=cms.InputTag("hltDeepInclusiveVertexFinderPFForPNetBTagFastSim"),
        ),
        clone_module(
            "hltDeepTrackVertexArbitratorPF",
            "hltDeepTrackVertexArbitratorPFForPNetBTagFastSim",
            primaryVertices=cms.InputTag(vertex_filter_name),
            secondaryVertices=cms.InputTag("hltDeepInclusiveSecondaryVerticesPFForPNetBTagFastSim"),
        ),
        clone_module(
            "hltDeepInclusiveMergedVerticesPF",
            "hltDeepInclusiveMergedVerticesPFForPNetBTagFastSim",
            secondaryVertices=cms.InputTag("hltDeepTrackVertexArbitratorPFForPNetBTagFastSim"),
        ),
        clone_module(
            "hltPrimaryVertexAssociation",
            "hltPrimaryVertexAssociationForPNetBTagFastSim",
            vertices=cms.InputTag(vertex_filter_name),
        ),
        clone_module(
            "hltParticleNetJetTagInfos",
            "hltParticleNetJetTagInfosForPNetBTagFastSim",
            vertices=cms.InputTag(vertex_filter_name),
            secondary_vertices=cms.InputTag("hltDeepInclusiveMergedVerticesPFForPNetBTagFastSim"),
            vertex_associator=cms.InputTag("hltPrimaryVertexAssociationForPNetBTagFastSim", "original"),
        ),
        clone_module(
            "hltParticleNetONNXJetTags",
            "hltParticleNetONNXJetTagsForPNetBTagFastSim",
            src=cms.InputTag("hltParticleNetJetTagInfosForPNetBTagFastSim"),
        ),
    )
    cloned_modules.extend(clone_name for clone_name in chain_clones if clone_name is not None)

    discriminator_name = clone_module(
        "hltParticleNetDiscriminatorsJetTags",
        "hltParticleNetDiscriminatorsJetTagsForPNetBTagFastSim",
    )
    if discriminator_name is None:
        return [], cloned_modules
    cloned_modules.append(discriminator_name)
    discriminator_module = getattr(process, discriminator_name)
    if hasattr(discriminator_module, "discriminators"):
        for discriminator in discriminator_module.discriminators:
            if hasattr(discriminator, "numerator"):
                discriminator.numerator = _retarget_vinput_tag(
                    discriminator.numerator,
                    "hltParticleNetONNXJetTags",
                    "hltParticleNetONNXJetTagsForPNetBTagFastSim",
                )
            if hasattr(discriminator, "denominator"):
                discriminator.denominator = _retarget_vinput_tag(
                    discriminator.denominator,
                    "hltParticleNetONNXJetTags",
                    "hltParticleNetONNXJetTagsForPNetBTagFastSim",
                )

    chain_replacements = (
        ("hltVerticesPFFilter", vertex_filter_name),
        ("hltDeepBLifetimeTagInfosPF", "hltDeepBLifetimeTagInfosPFForPNetBTagFastSim"),
        ("hltDeepInclusiveVertexFinderPF", "hltDeepInclusiveVertexFinderPFForPNetBTagFastSim"),
        ("hltDeepInclusiveSecondaryVerticesPF", "hltDeepInclusiveSecondaryVerticesPFForPNetBTagFastSim"),
        ("hltDeepTrackVertexArbitratorPF", "hltDeepTrackVertexArbitratorPFForPNetBTagFastSim"),
        ("hltDeepInclusiveMergedVerticesPF", "hltDeepInclusiveMergedVerticesPFForPNetBTagFastSim"),
        ("hltPrimaryVertexAssociation", "hltPrimaryVertexAssociationForPNetBTagFastSim"),
        ("hltParticleNetJetTagInfos", "hltParticleNetJetTagInfosForPNetBTagFastSim"),
        ("hltParticleNetONNXJetTags", "hltParticleNetONNXJetTagsForPNetBTagFastSim"),
        ("hltParticleNetDiscriminatorsJetTags", discriminator_name),
    )

    def clone_btag_filter_if_needed(module_name):
        if not hasattr(process, module_name):
            return None
        module = getattr(process, module_name)
        discriminator_parameters = [
            parameter_name
            for parameter_name in module.parameterNames_()
            if hasattr(getattr(module, parameter_name), "getModuleLabel")
            and hasattr(getattr(module, parameter_name), "getProductInstanceLabel")
            and _input_tag_parts(getattr(module, parameter_name))[0] == "hltParticleNetDiscriminatorsJetTags"
        ]
        if not discriminator_parameters:
            return None

        clone_name = module_name + "ForPNetBTagFastSim"
        if not hasattr(process, clone_name):
            clone = module.clone()
            for parameter_name in discriminator_parameters:
                setattr(
                    clone,
                    parameter_name,
                    _retarget_input_tag(
                        getattr(module, parameter_name),
                        "hltParticleNetDiscriminatorsJetTags",
                        discriminator_name,
                    ),
                )
            setattr(process, clone_name, clone)
            cloned_modules.append(clone_name)
        return clone_name

    changed_paths = []
    for path_name, path in process.paths_().items():
        if not _is_hlt_menu_path(path_name):
            continue
        if not _is_pnet_btag_vertex_path(path_name, include_tau_cross_paths):
            continue
        try:
            module_names = path.moduleNames()
        except Exception:
            continue

        btag_filter_replacements = []
        for module_name in module_names:
            clone_name = clone_btag_filter_if_needed(module_name)
            if clone_name is not None:
                btag_filter_replacements.append((module_name, clone_name))
        if not btag_filter_replacements:
            continue

        try:
            for old_name, new_name in chain_replacements:
                if old_name not in module_names or not hasattr(process, new_name):
                    continue
                path.replace(getattr(process, old_name), getattr(process, new_name))
            for old_name, new_name in btag_filter_replacements:
                path.replace(getattr(process, old_name), getattr(process, new_name))
            changed_paths.append(path_name)
        except Exception as exc:
            print("FastSim HLT: could not install PNet b-tag vertex chain in %s: %s" %
                  (path_name, exc))

    return sorted(changed_paths), sorted(set(cloned_modules))


def _l1_seed_modules_for_families(process, families):
    if not families:
        return set()

    selected_families = set(families)
    filters = process.filters_()
    seed_modules = set()
    for path_name, path in process.paths_().items():
        if not _is_hlt_menu_path(path_name):
            continue
        if _path_family(path_name) not in selected_families:
            continue

        try:
            module_names = path.moduleNames()
        except Exception:
            continue

        for module_name in module_names:
            module = filters.get(module_name)
            if module is not None and module.type_() == "HLTL1TSeed":
                seed_modules.add(module_name)

    return seed_modules


def _replace_l3_oi_seed_with_fast_tsg(process):
    if not hasattr(process, "hltIterL3OISeedsFromL2Muons"):
        print("FastSim HLT: hltIterL3OISeedsFromL2Muons not present; cannot install FastTSGFromL2Muon")
        return

    from RecoMuon.GlobalTrackingTools.MuonTrackingRegionCommon_cff import MuonTrackingRegionCommon

    process.hltIterL3OISeedsFromL2Muons = cms.EDProducer(
        "FastTSGFromL2Muon",
        MuonTrackingRegionBuilder=MuonTrackingRegionCommon.MuonTrackingRegionBuilder.clone(
            beamSpot=cms.InputTag("hltOnlineBeamSpot"),
            input=cms.InputTag("hltL2Muons", "UpdatedAtVtx"),
            maxRegions=cms.int32(5),
            Pt_min=cms.double(0.0),
        ),
        MuonCollectionLabel=cms.InputTag("hltL2Muons", "UpdatedAtVtx"),
        PtCut=cms.double(1.0),
        SeedCollectionLabels=cms.VInputTag(
            cms.InputTag("initialStepSeeds"),
            cms.InputTag("highPtTripletStepSeeds"),
            cms.InputTag("lowPtQuadStepSeeds"),
            cms.InputTag("lowPtTripletStepSeeds"),
            cms.InputTag("detachedQuadStepSeeds"),
            cms.InputTag("detachedTripletStepSeeds"),
            cms.InputTag("pixelPairStepSeeds"),
        ),
        SimTrackCollectionLabel=cms.InputTag("fastSimProducer"),
    )


def _replace_l3_oi_track_candidates_with_fastsim(process):
    if not hasattr(process, "hltIterL3OITrackCandidates"):
        print("FastSim HLT: hltIterL3OITrackCandidates not present; cannot install FastSim TrackCandidateProducer")
        return

    process.hltIterL3OITrackCandidates = cms.EDProducer(
        "TrackCandidateProducer",
        recHitCombinations=cms.InputTag("fastMatchedTrackerRecHitCombinations"),
        MinNumberOfCrossedLayers=cms.uint32(5),
        src=cms.InputTag("hltIterL3OISeedsFromL2Muons"),
        OverlapCleaning=cms.bool(False),
        SplitHits=cms.bool(False),
        simTracks=cms.InputTag("fastSimProducer"),
        propagator=cms.string("PropagatorWithMaterialOpposite"),
        maxSeedMatchEstimator=cms.untracked.double(200.0),
    )


def _install_fast_hlt_pixel_tracks(process, target_paths=(), scheduled_only=False):
    if not hasattr(process, "hltPixelTracksInRegionIter0L3Muon"):
        print("FastSim HLT: hltPixelTracksInRegionIter0L3Muon not present; cannot install FastSim HLT pixel tracks")
        return []

    old_region_selector = process.hltPixelTracksInRegionIter0L3Muon
    from RecoTracker.IterativeTracking.InitialStep_cff import (
        initialStepTrackingRegions as _initial_step_tracking_regions,
    )
    from RecoTracker.PixelTrackFitting.pixelFitterByHelixProjections_cfi import (
        pixelFitterByHelixProjections as _pixel_fitter_by_helix_projections,
    )
    from RecoTracker.PixelTrackFitting.pixelTrackFilterByKinematics_cfi import (
        pixelTrackFilterByKinematics as _pixel_track_filter_by_kinematics,
    )
    from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import RegionPSetBlock
    from FastSimulation.Tracking.TrajectorySeedProducer_cfi import (
        trajectorySeedProducer as _fastsim_trajectory_seed_producer,
    )

    process.hltPixelTripletTrackingRegions = _initial_step_tracking_regions.clone()
    process.hltPixelTripletSeeds = _fastsim_trajectory_seed_producer.clone(
        trackingRegions=cms.InputTag("hltPixelTripletTrackingRegions"),
        seedFinderSelector=_fastsim_seed_finder_selector(
            process,
            "hltPixelLayerTriplets",
            "hltElePixelHitDoubletsForTriplets",
        ),
    )
    process.hltPixelPairSeeds = _fastsim_trajectory_seed_producer.clone(
        trackingRegions=cms.InputTag("hltPixelTripletTrackingRegions"),
        seedFinderSelector=_fastsim_seed_finder_selector(
            process,
            "hltPixelLayerPairs",
            "hltElePixelHitDoublets",
        ),
    )
    process.hltSeedSequence = cms.Sequence(
        process.hltPixelTripletTrackingRegions
        + process.hltPixelTripletSeeds
        + process.hltPixelPairSeeds
    )
    process.hltPixelTracksFitterFastSim = _pixel_fitter_by_helix_projections.clone()
    process.hltPixelTracksFilterFastSim = _pixel_track_filter_by_kinematics.clone()
    process.hltPixelTracksFastSim = cms.EDProducer(
        "PixelTracksProducer",
        Fitter=cms.InputTag("hltPixelTracksFitterFastSim"),
        SeedProducer=cms.InputTag("hltPixelTripletSeeds"),
        RegionFactoryPSet=cms.PSet(
            RegionPSetBlock,
            ComponentName=cms.string("GlobalRegionProducer"),
        ),
        Filter=cms.InputTag("hltPixelTracksFilterFastSim"),
    )
    if hasattr(process.hltPixelTracksInRegionIter0L3Muon, "tracks"):
        process.hltPixelTracksInRegionIter0L3Muon.tracks = cms.InputTag("hltPixelTracksFastSim")

    required_modules = (
        "hltPixelTripletTrackingRegions",
        "hltPixelTripletSeeds",
        "hltPixelPairSeeds",
        "hltPixelTracksFitterFastSim",
        "hltPixelTracksFilterFastSim",
        "hltPixelTracksFastSim",
    )
    missing = [module_name for module_name in required_modules if not hasattr(process, module_name)]
    if missing:
        print("FastSim HLT: missing FastSim HLT pixel-track modules: %s" %
              ", ".join(missing))
        return []

    replacement = cms.Sequence(
        process.hltSeedSequence
        + process.hltPixelTracksFitterFastSim
        + process.hltPixelTracksFilterFastSim
        + process.hltPixelTracksFastSim
        + process.hltPixelTracksInRegionIter0L3Muon
    )

    changed_paths = []
    scheduled_paths = _scheduled_path_names(process) if scheduled_only else None
    for path_name, path in process.paths_().items():
        if not _is_hlt_menu_path(path_name):
            continue
        if scheduled_paths is not None and path_name not in scheduled_paths:
            continue
        if target_paths and not _path_requested(path_name, target_paths):
            continue
        try:
            module_names = path.moduleNames()
        except Exception:
            continue
        if "hltPixelTracksInRegionIter0L3Muon" not in module_names:
            continue
        try:
            path.replace(old_region_selector, replacement)
            changed_paths.append(path_name)
        except Exception as exc:
            print("FastSim HLT: could not splice FastSim hltPixelTracks sequence into %s: %s" %
                  (path_name, exc))

    if changed_paths:
        print("FastSim HLT: installed FastSim hltPixelTracks sequence in paths: %s" %
              _format_path_list(sorted(changed_paths)))

    return sorted(changed_paths)


def _layer_list_from_module(process, module_name):
    if not hasattr(process, module_name):
        return []

    module = getattr(process, module_name)
    if not hasattr(module, "layerList"):
        return []

    return [str(layer) for layer in module.layerList]


def _layer_pairs_from_module(process, module_name):
    if not hasattr(process, module_name):
        return []

    module = getattr(process, module_name)
    if not hasattr(module, "layerPairs"):
        return []

    return [int(layer_pair) for layer_pair in module.layerPairs]


def _fastsim_seed_finder_selector(process, layer_module_name, hit_pair_module_name):
    return cms.PSet(
        measurementTracker=cms.string(""),
        layerList=cms.vstring(*_layer_list_from_module(process, layer_module_name)),
        BPix=cms.PSet(
            TTRHBuilder=cms.string("WithoutRefit"),
            HitProducer=cms.string("TrackingRecHitProducer"),
        ),
        FPix=cms.PSet(
            TTRHBuilder=cms.string("WithoutRefit"),
            HitProducer=cms.string("TrackingRecHitProducer"),
        ),
        layerPairs=cms.vuint32(*_layer_pairs_from_module(process, hit_pair_module_name)),
    )


def _replace_egamma_gsf_pixel_seeds_with_fastsim(process):
    from FastSimulation.Tracking.TrajectorySeedProducer_cfi import (
        trajectorySeedProducer as _fastsim_trajectory_seed_producer,
    )

    replacements = (
        (
            "hltElePixelSeedsDoublets",
            "hltEleSeedsTrackingRegions",
            "hltPixelLayerPairs",
            "hltElePixelHitDoublets",
        ),
        (
            "hltElePixelSeedsTriplets",
            "hltEleSeedsTrackingRegions",
            "hltPixelLayerTriplets",
            "hltElePixelHitDoubletsForTriplets",
        ),
        (
            "hltElePixelSeedsDoubletsUnseeded",
            "hltEleSeedsTrackingRegionsUnseeded",
            "hltPixelLayerPairs",
            "hltElePixelHitDoubletsUnseeded",
        ),
        (
            "hltElePixelSeedsTripletsUnseeded",
            "hltEleSeedsTrackingRegionsUnseeded",
            "hltPixelLayerTriplets",
            "hltElePixelHitDoubletsForTripletsUnseeded",
        ),
    )

    installed_modules = []
    for seed_module_name, region_module_name, layer_module_name, hit_pair_module_name in replacements:
        missing = [
            module_name
            for module_name in (seed_module_name, region_module_name, layer_module_name)
            if not hasattr(process, module_name)
        ]
        if missing:
            print("FastSim HLT: skipped EGamma GSF seed replacement for %s; missing %s" %
                  (seed_module_name, ", ".join(missing)))
            continue

        layer_list = _layer_list_from_module(process, layer_module_name)
        if not layer_list:
            print("FastSim HLT: skipped EGamma GSF seed replacement for %s; %s has no layerList" %
                  (seed_module_name, layer_module_name))
            continue

        setattr(
            process,
            seed_module_name,
            _fastsim_trajectory_seed_producer.clone(
                trackingRegions=cms.InputTag(region_module_name),
                seedFinderSelector=_fastsim_seed_finder_selector(
                    process,
                    layer_module_name,
                    hit_pair_module_name,
                ),
            ),
        )
        installed_modules.append(seed_module_name)

    if installed_modules:
        print("FastSim HLT: replaced EGamma GSF pixel seed modules with FastSim TrajectorySeedProducer: %s" %
              _format_path_list(sorted(installed_modules)))

    return installed_modules


def _replace_egamma_gsf_track_candidates_with_fastsim(process):
    from FastSimulation.Tracking.TrackCandidateProducer_cfi import (
        trackCandidateProducer as _fastsim_track_candidate_producer,
    )

    replacements = (
        ("hltEgammaCkfTrackCandidatesForGSF", "hltEgammaElectronPixelSeeds"),
        ("hltEgammaCkfTrackCandidatesForGSFUnseeded", "hltEgammaElectronPixelSeedsUnseeded"),
    )

    installed_modules = []
    for track_candidate_module_name, seed_module_name in replacements:
        if not hasattr(process, track_candidate_module_name):
            continue

        setattr(
            process,
            track_candidate_module_name,
            _fastsim_track_candidate_producer.clone(
                src=cms.InputTag(seed_module_name),
                MinNumberOfCrossedLayers=cms.uint32(5),
                OverlapCleaning=cms.bool(True),
            ),
        )
        installed_modules.append(track_candidate_module_name)

    if installed_modules:
        print("FastSim HLT: replaced EGamma GSF track-candidate modules with FastSim TrackCandidateProducer: %s" %
              _format_path_list(sorted(installed_modules)))

    return installed_modules


def _keep_fast_l3_muon_seed_debug_products(process):
    commands = (
        "keep *_hltIterL3OISeedsFromL2Muons_*_*",
        "keep *_initialStepSeeds_*_*",
        "keep *_highPtTripletStepSeeds_*_*",
        "keep *_lowPtQuadStepSeeds_*_*",
        "keep *_lowPtTripletStepSeeds_*_*",
        "keep *_detachedQuadStepSeeds_*_*",
        "keep *_detachedTripletStepSeeds_*_*",
        "keep *_pixelPairStepSeeds_*_*",
        "keep *_hltIterL3OITrackCandidates_*_*",
    )

    for output_module in process.outputModules_().values():
        if not hasattr(output_module, "outputCommands"):
            continue
        if output_module.type_() == "NanoAODOutputModule":
            continue

        existing = set(output_module.outputCommands)
        for command in commands:
            if command not in existing:
                output_module.outputCommands.append(command)


def _keep_hlt_debug_products(process):
    commands = (
        "keep *_hltGtStage2Digis_*_*",
        "keep *_hltL1MuonsPt0_*_*",
        "keep *_hltL2MuonCandidates_*_*",
        "keep *_hltL2Muons_*_*",
        "keep *_hltIterL3MuonCandidates_*_*",
        "keep *_hltIterL3Muons_*_*",
        "keep *_hltIterL3MuonsNoID_*_*",
        "keep *_hltIterL3OIMuonTrackSelectionHighPurity_*_*",
        "keep *_hltIterL3MuonMerged_*_*",
        "keep *_hltIterL3MuonAndMuonFromL1Merged_*_*",
        "keep *_hltIter0L3Muon*_*_*",
        "keep *_hltPixelTracksInRegionIter0L3Muon_*_*",
        "keep *_hltPixelTracks_*_*",
        "keep *_hltPixelTracksFastSim_*_*",
        "keep *_hltPixelTripletTrackingRegions_*_*",
        "keep *_hltPixelTripletSeeds_*_*",
        "keep *_hltPixelPairSeeds_*_*",
        "keep *_hltPixelVertices_*_*",
        "keep *_hltOnlineBeamSpot_*_*",
        "keep *_hltMeasurementTrackerEvent_*_*",
        "keep *_hltSiPixelClusters_*_*",
        "keep *_hltSiPixelRecHits_*_*",
        "keep *_hltIter0L3MuonTrackSelectionHighPurity_*_*",
        "keep *_hltIter0L3MuonCtfWithMaterialTracks_*_*",
        "keep *_hltL3MuonRelTrkIsolationVVL_*_*",
        "keep *_hltL3MuonCombRelIsolationVVVL_*_*",
        "keep *_hltL3MuonOpenRelTrkIsolationVVL_*_*",
        "keep *_hltMuonTkRelIsolationCut0p14Map_*_*",
        "keep *_hltMuonTkRelIsolationCut0p3Map_*_*",
        "keep *_hltMuonTkRelIsolationCut0p09MapNoVtx_*_*",
        "keep *_hltEgammaCandidates_*_*",
        "keep *_hltEgammaSuperClustersToPixelMatch_*_*",
        "keep *_hltEgammaElectronPixelSeeds_*_*",
        "keep *_hltEgammaCkfTrackCandidatesForGSF_*_*",
        "keep *_hltEgammaGsfTracks_*_*",
        "keep *_hltEgammaGsfTrackVars_*_*",
        "keep *_hltEgammaGsfElectrons_*_*",
        "keep *_hltEgammaPixelMatchVars_*_*",
        "keep *_hltElePixelSeeds*_*_*",
        "keep *_hltElePixelHit*_*_*",
        "keep *_fastMatchedTrackerRecHitCombinations_*_*",
        "keep *_hltEle115*_*_*",
        "keep *_hltEle135*_*_*",
        "keep *_hltEG115*_*_*",
        "keep *_hltEG135*_*_*",
        "keep *_hltAK4PFJets_*_*",
        "keep *_hltPFMETProducer_*_*",
        "keep *_hltHpsPFTauProducer_*_*",
        "keep *_hltDeepCombinedSecondaryVertexBJetTagsPF_*_*",
        "keep *_hltScoutingMuonPacker_*_*",
    )

    for output_module in process.outputModules_().values():
        if not hasattr(output_module, "outputCommands"):
            continue
        if output_module.type_() == "NanoAODOutputModule":
            continue

        existing = set(output_module.outputCommands)
        for command in commands:
            if command not in existing:
                output_module.outputCommands.append(command)


def _customizeFastSimHLT(process, force_review_default=False):
    """Apply local compatibility fixes for running modern HLT menus in FastSim."""
    profiler = _CustomizerProfiler(_configured_profile_customizer())
    keep_paths = _configured_keep_paths()
    only_paths = _configured_only_paths()
    keep_l1_seed_modules = _configured_keep_l1_seed_modules()
    keep_l1_seed_families = _configured_keep_l1_seed_families()
    enable_review_default = force_review_default or _configured_enable_review_default()
    enable_muon_prototype = _configured_enable_muon_prototype(enable_review_default)
    enable_muon_iso_prototype = _configured_enable_muon_iso_prototype(enable_review_default)
    enable_muon_pixel_tracks_prototype = _configured_enable_muon_pixel_tracks_prototype(
        enable_muon_iso_prototype
    )
    enable_object_prototypes = _configured_enable_object_prototypes(enable_review_default)
    enable_egamma_gsf_prototype = _configured_enable_egamma_gsf_prototype(enable_review_default)
    enable_egamma_mkfit_iso_prototype = _configured_enable_egamma_mkfit_iso_prototype(enable_review_default)
    enable_pfmet_mkfit_prototype = _configured_enable_pfmet_mkfit_prototype(enable_review_default)
    enable_high_pfmet_prototype = _configured_enable_high_pfmet_prototype(enable_review_default)
    enable_pfjet_pfht_mkfit_prototype = _configured_enable_pfjet_pfht_mkfit_prototype(enable_review_default)
    enable_ak8_softdrop_prototype = _configured_enable_ak8_softdrop_prototype(enable_review_default)
    enable_cross_object_pf_mkfit_prototype = _configured_enable_cross_object_pf_mkfit_prototype(enable_review_default)
    enable_pnet_vertex_prototype = _configured_enable_pnet_vertex_prototype(False)
    enable_pnet_btag_vertex_prototype = _configured_enable_pnet_btag_vertex_prototype(False)
    enable_jet_met_btag_l1_prototypes = _configured_enable_jet_met_btag_l1_prototypes(
        enable_object_prototypes or
        enable_pfmet_mkfit_prototype or
        enable_high_pfmet_prototype or
        enable_pfjet_pfht_mkfit_prototype or
        enable_ak8_softdrop_prototype or
        enable_cross_object_pf_mkfit_prototype or
        enable_pnet_vertex_prototype
    )
    enable_egamma_tau_l1_prototypes = _configured_enable_egamma_tau_l1_prototypes(enable_review_default)
    pnet_vertex_source = _configured_pnet_vertex_source()
    pnet_vertex_cut = _configured_pnet_vertex_cut()
    pnet_vertex_track_source = _configured_pnet_vertex_track_source()
    pnet_btag_vertex_source = _configured_pnet_btag_vertex_source()
    pnet_btag_vertex_cut = _configured_pnet_btag_vertex_cut()
    pnet_btag_vertex_include_tau_cross_paths = _configured_pnet_btag_vertex_include_tau_cross_paths()
    object_l1_save_tags = _configured_object_l1_save_tags()
    relax_l3_muon_id = _configured_relax_l3_muon_id()
    use_fast_l3_muon_seeds = _configured_use_fast_l3_muon_seeds(enable_muon_prototype)
    use_fast_l3_muon_track_candidates = _configured_use_fast_l3_muon_track_candidates(enable_muon_prototype)
    muon_iso_track_source = _configured_muon_iso_track_source()
    muon_iso_track_quality = _configured_muon_iso_track_quality()
    if enable_muon_iso_prototype and not muon_iso_track_quality:
        muon_iso_track_quality = "loose"
    keep_debug_products = _configured_keep_debug_products()
    use_combined_pruner = _configured_use_combined_pruner(enable_review_default)
    profiler.mark("read switches")
    object_prototype_l1_seed_modules = set()
    jet_met_btag_l1_seed_modules = set()
    egamma_tau_l1_seed_modules = set()
    cross_object_pf_l1_seed_modules = set()
    object_l1_seed_modules_for_save_tags_control = set()
    if enable_muon_prototype:
        keep_l1_seed_modules.update(_DEFAULT_MUON_L1_SEED_MODULES)
        print("FastSim HLT: enabling muon prototype mode")
    if enable_object_prototypes:
        object_prototype_l1_seed_modules.update(_DEFAULT_OBJECT_L1_SEED_MODULES)
        keep_l1_seed_modules.update(object_prototype_l1_seed_modules)
        object_l1_seed_modules_for_save_tags_control.update(_DEFAULT_OBJECT_L1_SEED_MODULES)
        print("FastSim HLT: enabling curated object L1 seed prototype mode")
    if enable_jet_met_btag_l1_prototypes:
        jet_met_btag_l1_seed_modules.update(_DEFAULT_JET_MET_BTAG_L1_SEED_MODULES)
        keep_l1_seed_modules.update(jet_met_btag_l1_seed_modules)
        print("FastSim HLT: enabling active jet/MET/b-tag L1 seed prototype mode")
    if enable_egamma_tau_l1_prototypes:
        egamma_tau_l1_seed_modules.update(_DEFAULT_EGAMMA_TAU_L1_SEED_MODULES)
        keep_l1_seed_modules.update(egamma_tau_l1_seed_modules)
        print("FastSim HLT: enabling active EGamma/tau L1 seed prototype mode")
    family_l1_seed_modules = _l1_seed_modules_for_families(process, keep_l1_seed_families)
    keep_l1_seed_modules.update(family_l1_seed_modules)
    pfmet_mkfit_prototype_paths = []
    if enable_pfmet_mkfit_prototype:
        pfmet_mkfit_prototype_paths = _pfmet_mkfit_prototype_paths(process)
        keep_paths.update(pfmet_mkfit_prototype_paths)
        keep_l1_seed_modules.update(_DEFAULT_OBJECT_L1_SEED_MODULES)
        print("FastSim HLT: enabling PF-MET/PFMHT mkFit prototype mode")
    high_pfmet_prototype_paths = []
    if enable_high_pfmet_prototype:
        high_pfmet_prototype_paths = _high_pfmet_prototype_paths(process)
        keep_paths.update(high_pfmet_prototype_paths)
        keep_l1_seed_modules.update(_DEFAULT_OBJECT_L1_SEED_MODULES)
        print("FastSim HLT: enabling high-PFMET prototype mode")
    pfjet_pfht_mkfit_prototype_paths = []
    if enable_pfjet_pfht_mkfit_prototype:
        pfjet_pfht_mkfit_prototype_paths = _pfjet_pfht_mkfit_prototype_paths(process)
        keep_paths.update(pfjet_pfht_mkfit_prototype_paths)
        keep_l1_seed_modules.update(_DEFAULT_OBJECT_L1_SEED_MODULES)
        print("FastSim HLT: enabling PFJet/PFHT mkFit prototype mode")
    ak8_softdrop_prototype_paths = []
    if enable_ak8_softdrop_prototype:
        ak8_softdrop_prototype_paths = _ak8_softdrop_prototype_paths(process)
        keep_paths.update(ak8_softdrop_prototype_paths)
        keep_l1_seed_modules.update(_DEFAULT_OBJECT_L1_SEED_MODULES)
        print("FastSim HLT: enabling AK8 soft-drop prototype mode")
    cross_object_pf_mkfit_prototype_paths = []
    if enable_cross_object_pf_mkfit_prototype:
        cross_object_pf_mkfit_prototype_paths = _cross_object_pf_mkfit_prototype_paths(process)
        keep_paths.update(cross_object_pf_mkfit_prototype_paths)
        cross_object_pf_l1_seed_modules.update(_DEFAULT_CROSS_OBJECT_PF_L1_SEED_MODULES)
        keep_l1_seed_modules.update(_DEFAULT_MUON_L1_SEED_MODULES)
        keep_l1_seed_modules.update(_DEFAULT_OBJECT_L1_SEED_MODULES)
        keep_l1_seed_modules.update(_DEFAULT_EGAMMA_TAU_L1_SEED_MODULES)
        keep_l1_seed_modules.update(cross_object_pf_l1_seed_modules)
        print("FastSim HLT: enabling cross-object PF mkFit prototype mode")
    egamma_mkfit_iso_prototype_paths = []
    if enable_egamma_mkfit_iso_prototype:
        egamma_mkfit_iso_prototype_paths = _egamma_mkfit_iso_prototype_paths(process)
        keep_paths.update(egamma_mkfit_iso_prototype_paths)
        keep_l1_seed_modules.update(_DEFAULT_EGAMMA_TAU_L1_SEED_MODULES)
        print("FastSim HLT: enabling WPTight/TightIso EGamma mkFit-isolation prototype mode")
    pnet_btag_tau_prototype_paths = []
    if enable_pnet_vertex_prototype or enable_pnet_btag_vertex_prototype:
        pnet_btag_tau_prototype_paths = _pnet_btag_tau_prototype_paths(process)
        keep_paths.update(pnet_btag_tau_prototype_paths)
        keep_l1_seed_modules.update(_DEFAULT_OBJECT_L1_SEED_MODULES)
        keep_l1_seed_modules.update(_DEFAULT_JET_MET_BTAG_L1_SEED_MODULES)
        keep_l1_seed_modules.update(_DEFAULT_PNET_BTAG_TAU_L1_SEED_MODULES)
        print("FastSim HLT: enabling curated PNet b-tag/tau prototype paths")
    profiler.mark("select requested prototypes")

    _set_audit_bool(process, "muonPrototypeEnabled", enable_muon_prototype)
    _set_audit_bool(process, "muonIsoPrototypeEnabled", enable_muon_iso_prototype)
    _set_audit_bool(process, "muonPixelTracksPrototypeEnabled", enable_muon_pixel_tracks_prototype)
    _set_audit_bool(process, "reviewDefaultEnabled", enable_review_default)
    _set_audit_bool(process, "objectPrototypesEnabled", enable_object_prototypes)
    _set_audit_bool(process, "jetMetBTagL1PrototypesEnabled", enable_jet_met_btag_l1_prototypes)
    _set_audit_bool(process, "egammaTauL1PrototypesEnabled", enable_egamma_tau_l1_prototypes)
    _set_audit_bool(process, "egammaGsfPrototypeEnabled", enable_egamma_gsf_prototype)
    _set_audit_bool(process, "egammaMkFitIsoPrototypeEnabled", enable_egamma_mkfit_iso_prototype)
    _set_audit_bool(process, "pfmetMkFitPrototypeEnabled", enable_pfmet_mkfit_prototype)
    _set_audit_bool(process, "highPfmetPrototypeEnabled", enable_high_pfmet_prototype)
    _set_audit_bool(process, "pfjetPfhtMkFitPrototypeEnabled", enable_pfjet_pfht_mkfit_prototype)
    _set_audit_bool(process, "ak8SoftDropPrototypeEnabled", enable_ak8_softdrop_prototype)
    _set_audit_bool(process, "crossObjectPfMkFitPrototypeEnabled", enable_cross_object_pf_mkfit_prototype)
    _set_audit_bool(process, "pnetVertexPrototypeEnabled", enable_pnet_vertex_prototype)
    _set_audit_bool(process, "pnetBTagVertexPrototypeEnabled", enable_pnet_btag_vertex_prototype)
    _set_audit_bool(
        process,
        "pnetBTagVertexIncludeTauCrossPaths",
        pnet_btag_vertex_include_tau_cross_paths,
    )
    _set_audit_bool(process, "objectL1SaveTagsEnabled", object_l1_save_tags)
    _set_audit_bool(process, "fastL3MuonSeedsEnabled", use_fast_l3_muon_seeds)
    _set_audit_bool(process, "fastL3MuonTrackCandidatesEnabled", use_fast_l3_muon_track_candidates)
    _set_audit_bool(process, "relaxedL3MuonIdEnabled", relax_l3_muon_id)
    _set_audit_bool(process, "debugProductsKept", keep_debug_products)
    _set_audit_bool(process, "combinedPrunerEnabled", use_combined_pruner)
    _set_audit_vstring(process, "muonIsoConfiguredTrackSource", [muon_iso_track_source])
    _set_audit_vstring(process, "requestedKeepPaths", sorted(keep_paths))
    _set_audit_vstring(process, "requestedOnlyPaths", sorted(only_paths))
    _set_audit_vstring(
        process,
        "objectPrototypeL1SeedModules",
        sorted(object_prototype_l1_seed_modules),
    )
    _set_audit_vstring(
        process,
        "jetMetBTagPrototypeL1SeedModules",
        sorted(jet_met_btag_l1_seed_modules),
    )
    _set_audit_vstring(
        process,
        "egammaTauPrototypeL1SeedModules",
        sorted(egamma_tau_l1_seed_modules),
    )
    _set_audit_vstring(
        process,
        "crossObjectPfPrototypeL1SeedModules",
        sorted(cross_object_pf_l1_seed_modules),
    )
    _set_audit_vstring(
        process,
        "unsupportedEgammaTauL1SeedModules",
        _UNSUPPORTED_EGAMMA_TAU_L1_SEED_MODULES,
    )
    _set_audit_vstring(
        process,
        "unsupportedJetMetBTagTauL1SeedModules",
        _UNSUPPORTED_JET_MET_BTAG_TAU_L1_SEED_MODULES,
    )
    _set_audit_vstring(process, "requestedKeepL1SeedFamilies", sorted(keep_l1_seed_families))
    _set_audit_vstring(process, "familySelectedL1SeedModules", sorted(family_l1_seed_modules))
    _set_audit_vstring(process, "requestedKeepL1SeedModules", sorted(keep_l1_seed_modules))
    _set_audit_vstring(process, "pfmetMkFitPrototypePaths", pfmet_mkfit_prototype_paths)
    _set_audit_vstring(process, "highPfmetPrototypePaths", high_pfmet_prototype_paths)
    _set_audit_vstring(process, "pfjetPfhtMkFitPrototypePaths", pfjet_pfht_mkfit_prototype_paths)
    _set_audit_vstring(process, "ak8SoftDropPrototypePaths", ak8_softdrop_prototype_paths)
    _set_audit_vstring(process, "crossObjectPfMkFitPrototypePaths", cross_object_pf_mkfit_prototype_paths)
    _set_audit_vstring(process, "egammaMkFitIsoPrototypePaths", egamma_mkfit_iso_prototype_paths)
    _set_audit_vstring(process, "pnetBTagTauPrototypePaths", pnet_btag_tau_prototype_paths)
    _set_audit_vstring(process, "muonPixelTracksPrototypePaths", [])
    _set_audit_vstring(process, "pnetVertexConfiguredTrackSource", [pnet_vertex_track_source])
    _set_audit_vstring(process, "pnetVertexConfiguredSource", [pnet_vertex_source])
    _set_audit_vstring(process, "pnetVertexConfiguredCut", [pnet_vertex_cut])
    _set_audit_vstring(process, "pnetBTagVertexConfiguredSource", [pnet_btag_vertex_source])
    _set_audit_vstring(process, "pnetBTagVertexConfiguredCut", [pnet_btag_vertex_cut])
    _set_audit_vstring(process, "muonIsoConfiguredTrackQuality", [muon_iso_track_quality])
    profiler.mark("write audit prescan state")

    if only_paths:
        keep_paths = keep_paths.union(only_paths)
        print("FastSim HLT: running only requested HLT menu paths: %s" %
              _format_path_list(sorted(only_paths)))
    if keep_paths:
        print("FastSim HLT: keeping requested paths during pruning: %s" %
              _format_path_list(sorted(keep_paths)))
    if enable_egamma_mkfit_iso_prototype:
        print("FastSim HLT: keeping WPTight/TightIso EGamma mkFit-isolation prototype paths: %s" %
              _format_path_list(egamma_mkfit_iso_prototype_paths))
    if keep_l1_seed_modules:
        print("FastSim HLT: preserving requested L1 seed expressions: %s" %
              _format_path_list(sorted(keep_l1_seed_modules)))
    if keep_l1_seed_families:
        print("FastSim HLT: preserving L1 seed expressions by path family %s: %s" %
              (_format_path_list(sorted(keep_l1_seed_families)),
               _format_path_list(sorted(family_l1_seed_modules))))
    if relax_l3_muon_id:
        print("FastSim HLT: relaxing L3 muon trigger ID filters")
    if use_fast_l3_muon_seeds:
        print("FastSim HLT: replacing hltIterL3OISeedsFromL2Muons with FastTSGFromL2Muon")
    if use_fast_l3_muon_track_candidates:
        print("FastSim HLT: replacing hltIterL3OITrackCandidates with FastSim TrackCandidateProducer")
    if enable_muon_iso_prototype:
        print("FastSim HLT: enabling L3 muon track-isolation prototype mode")
    if enable_muon_pixel_tracks_prototype:
        print("FastSim HLT: enabling FastSim HLT pixel-track prototype mode")
    if muon_iso_track_quality:
        print("FastSim HLT: setting L3 muon isolation track-selection quality to %s" %
              muon_iso_track_quality)
    if enable_egamma_gsf_prototype:
        print("FastSim HLT: enabling EGamma GSF-track prototype mode")
    if enable_pfmet_mkfit_prototype:
        print("FastSim HLT: keeping PF-MET/PFMHT prototype paths: %s" %
              _format_path_list(pfmet_mkfit_prototype_paths))
    if enable_high_pfmet_prototype:
        print("FastSim HLT: keeping high-PFMET prototype paths: %s" %
              _format_path_list(high_pfmet_prototype_paths))
    if enable_pfjet_pfht_mkfit_prototype:
        print("FastSim HLT: keeping PFJet/PFHT prototype paths: %s" %
              _format_path_list(pfjet_pfht_mkfit_prototype_paths))
    if enable_ak8_softdrop_prototype:
        print("FastSim HLT: keeping AK8 soft-drop prototype paths: %s" %
              _format_path_list(ak8_softdrop_prototype_paths))
    if enable_cross_object_pf_mkfit_prototype:
        print("FastSim HLT: keeping cross-object PF prototype paths: %s" %
              _format_path_list(cross_object_pf_mkfit_prototype_paths))
    if pnet_btag_tau_prototype_paths:
        print("FastSim HLT: keeping curated PNet b-tag/tau prototype paths: %s" %
              _format_path_list(pnet_btag_tau_prototype_paths))
    if enable_pnet_vertex_prototype:
        print("FastSim HLT: enabling PNet vertex compatibility prototype mode "
              "(trackSource=%s, source=%s, cut=%s)" %
              (pnet_vertex_track_source or "<unchanged>", pnet_vertex_source, pnet_vertex_cut))
    if enable_pnet_btag_vertex_prototype:
        print("FastSim HLT: enabling PNet b-tag path-local vertex prototype mode "
              "(source=%s, cut=%s, includeTauCrossPaths=%s)" %
              (
                  pnet_btag_vertex_source,
                  pnet_btag_vertex_cut,
                  pnet_btag_vertex_include_tau_cross_paths,
              ))
    if keep_debug_products:
        print("FastSim HLT: keeping diagnostic HLT products in PoolOutputModules")
    if use_combined_pruner:
        print("FastSim HLT: using combined scheduled-path pruner")
    if enable_review_default:
        print("FastSim HLT: enabling review default path set")
    profiler.mark("print configuration summary")

    process.load("Geometry.GEMGeometryBuilder.gemGeometryDB_cfi")
    process.load("RecoLocalCalo.EcalRecProducers.alpaka_serial_syncEcalMultifitConditionsHostESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecAlgos.alpaka_serial_syncHcalRecoParamWithPulseShapeESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.alpaka_serial_syncHcalMahiConditionsESProducer_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.alpaka_serial_syncHcalSiPMCharacteristicsESProducer_cfi")
    process.load("RecoTracker.MkFit.mkFitGeometryESProducer_cfi")
    if (
        enable_pfmet_mkfit_prototype or
        enable_high_pfmet_prototype or
        enable_pfjet_pfht_mkfit_prototype or
        enable_ak8_softdrop_prototype or
        enable_cross_object_pf_mkfit_prototype
    ):
        _add_direct_mkfit_geometry_esproducer(process)

    process.es_prefer_alpaka_serial_syncEcalMultifitConditionsHostESProducer = cms.ESPrefer(
        "alpaka_serial_sync::EcalMultifitConditionsHostESProducer",
        "alpaka_serial_syncEcalMultifitConditionsHostESProducer",
    )
    process.es_prefer_alpaka_serial_syncHcalRecoParamWithPulseShapeESProducer = cms.ESPrefer(
        "alpaka_serial_sync::HcalRecoParamWithPulseShapeESProducer",
        "alpaka_serial_syncHcalRecoParamWithPulseShapeESProducer",
    )
    process.es_prefer_alpaka_serial_syncHcalMahiConditionsESProducer = cms.ESPrefer(
        "alpaka_serial_sync::HcalMahiConditionsESProducer",
        "alpaka_serial_syncHcalMahiConditionsESProducer",
    )
    process.es_prefer_alpaka_serial_syncHcalSiPMCharacteristicsESProducer = cms.ESPrefer(
        "alpaka_serial_sync::HcalSiPMCharacteristicsESProducer",
        "alpaka_serial_syncHcalSiPMCharacteristicsESProducer",
    )
    profiler.mark("load EventSetup compatibility")

    if hasattr(process, "hltGemRecHits"):
        process.hltGemRecHits.applyMasking = cms.bool(False)

    preserved_l1_seed_modules = []
    object_l1_seed_modules_with_save_tags_disabled = []
    rewritten_l1_seed_modules = []
    for module_name, module in process.filters_().items():
        if module.type_() == "HLTL1TSeed":
            if module_name in keep_l1_seed_modules:
                if (
                    module_name in object_l1_seed_modules_for_save_tags_control and
                    module_name not in _DEFAULT_MUON_L1_SEED_MODULES and
                    not object_l1_save_tags and
                    hasattr(module, "saveTags")
                ):
                    module.saveTags = cms.bool(False)
                    object_l1_seed_modules_with_save_tags_disabled.append(module_name)
                preserved_l1_seed_modules.append(module_name)
                continue

            module.L1SeedsLogicalExpression = cms.string("L1GlobalDecision")
            rewritten_l1_seed_modules.append(module_name)
        elif module.type_() == "TriggerResultsFilter" and hasattr(module, "throw"):
            module.throw = cms.bool(False)

    _set_audit_vstring(process, "preservedL1SeedModules", sorted(preserved_l1_seed_modules))
    _set_audit_vstring(
        process,
        "objectL1SeedModulesWithSaveTagsDisabled",
        sorted(object_l1_seed_modules_with_save_tags_disabled),
    )
    _set_audit_vstring(process, "rewrittenL1SeedModules", sorted(rewritten_l1_seed_modules))
    profiler.mark("rewrite L1 seed expressions")

    if relax_l3_muon_id:
        for module in process.producers_().values():
            if module.type_() == "MuonIDFilterProducerForHLT" and hasattr(module, "applyTriggerIdLoose"):
                module.applyTriggerIdLoose = cms.bool(False)

    _modify_if_present(process, "hltL3TrajSeedOIHit", modify_hltL3TrajSeedOIHit)
    _modify_if_present(process, "hltL3NoFiltersNoVtxTrajSeedOIHit", modify_hltL3TrajSeedOIHit)
    _modify_if_present(process, "hltL3TrajSeedIOHit", modify_hltL3TrajSeedIOHit)
    _modify_if_present(process, "hltL3NoFiltersNoVtxTrajSeedIOHit", modify_hltL3TrajSeedIOHit)
    profiler.mark("apply L3 muon hit customizations")

    if use_fast_l3_muon_seeds:
        _replace_l3_oi_seed_with_fast_tsg(process)
        _keep_fast_l3_muon_seed_debug_products(process)
    if use_fast_l3_muon_track_candidates:
        _replace_l3_oi_track_candidates_with_fastsim(process)
        _keep_fast_l3_muon_seed_debug_products(process)
    if enable_egamma_gsf_prototype:
        egamma_gsf_fastsim_seed_modules = _replace_egamma_gsf_pixel_seeds_with_fastsim(process)
        egamma_gsf_fastsim_track_candidate_modules = _replace_egamma_gsf_track_candidates_with_fastsim(process)
    else:
        egamma_gsf_fastsim_seed_modules = []
        egamma_gsf_fastsim_track_candidate_modules = []
    profiler.mark("install tracking substitutions")
    if enable_pfmet_mkfit_prototype or enable_high_pfmet_prototype or enable_cross_object_pf_mkfit_prototype:
        pfmet_mht_rerouted_modules = _reroute_pfmet_mht_to_corrected_pfjets(process)
        if pfmet_mht_rerouted_modules:
            print("FastSim HLT: rerouted PF-MHT jet inputs away from FastSim-hostile PFJet ID collections: %s" %
                  _format_path_list(pfmet_mht_rerouted_modules))
    else:
        pfmet_mht_rerouted_modules = []
    if enable_pfjet_pfht_mkfit_prototype or enable_ak8_softdrop_prototype or enable_cross_object_pf_mkfit_prototype:
        pfjet_id_filter_rerouted_modules = _reroute_pfjet_id_filters_to_corrected_pfjets(process)
        if pfjet_id_filter_rerouted_modules:
            print("FastSim HLT: rerouted PFJet ID filter inputs to corrected PF jets: %s" %
                  _format_path_list(pfjet_id_filter_rerouted_modules))
    else:
        pfjet_id_filter_rerouted_modules = []
    muon_iso_track_rerouted_modules = _reroute_muon_iso_track_sources(process, muon_iso_track_source)
    if muon_iso_track_rerouted_modules:
        print("FastSim HLT: rerouted muon track-isolation inputs: %s" %
              _format_path_list(muon_iso_track_rerouted_modules))
    muon_iso_track_quality_modules = _set_muon_iso_track_selection_quality(
        process,
        muon_iso_track_quality,
    )
    if muon_iso_track_quality_modules:
        print("FastSim HLT: changed L3 muon isolation track-selection quality: %s" %
              _format_path_list(muon_iso_track_quality_modules))
    if enable_pnet_vertex_prototype:
        pnet_vertex_relaxed_modules = _relax_pnet_vertex_filters_for_fastsim(
            process,
            pnet_vertex_source,
            pnet_vertex_cut,
            pnet_vertex_track_source,
        )
        if pnet_vertex_relaxed_modules:
            print("FastSim HLT: relaxed PNet vertex filters for FastSim: %s" %
                  _format_path_list(pnet_vertex_relaxed_modules))
    else:
        pnet_vertex_relaxed_modules = []
    if enable_pnet_btag_vertex_prototype:
        (
            pnet_btag_vertex_paths,
            pnet_btag_vertex_modules,
        ) = _install_pnet_btag_vertex_filter_for_fastsim(
            process,
            pnet_btag_vertex_source,
            pnet_btag_vertex_cut,
            pnet_btag_vertex_include_tau_cross_paths,
        )
        if pnet_btag_vertex_paths:
            print("FastSim HLT: installed path-local PNet b-tag vertex chain in paths: %s" %
                  _format_path_list(pnet_btag_vertex_paths))
    else:
        pnet_btag_vertex_paths = []
        pnet_btag_vertex_modules = []
    profiler.mark("reroute PF and vertex inputs")
    _set_audit_vstring(
        process,
        "egammaGsfFastSimSeedModules",
        sorted(egamma_gsf_fastsim_seed_modules),
    )
    _set_audit_vstring(
        process,
        "egammaGsfFastSimTrackCandidateModules",
        sorted(egamma_gsf_fastsim_track_candidate_modules),
    )
    _set_audit_vstring(
        process,
        "pfmetMhtReroutedModules",
        sorted(pfmet_mht_rerouted_modules),
    )
    _set_audit_vstring(
        process,
        "pfJetIdFilterReroutedModules",
        sorted(pfjet_id_filter_rerouted_modules),
    )
    _set_audit_vstring(
        process,
        "muonIsoTrackReroutedModules",
        sorted(muon_iso_track_rerouted_modules),
    )
    _set_audit_vstring(
        process,
        "pnetVertexRelaxedModules",
        sorted(pnet_vertex_relaxed_modules),
    )
    _set_audit_vstring(
        process,
        "pnetBTagVertexPrototypePaths",
        sorted(pnet_btag_vertex_paths),
    )
    _set_audit_vstring(
        process,
        "pnetBTagVertexPrototypeModules",
        sorted(pnet_btag_vertex_modules),
    )
    if keep_debug_products:
        _keep_hlt_debug_products(process)
    profiler.mark("write audit postscan state")

    if only_paths:
        removed_unrequested = _remove_unrequested_hlt_menu_paths(process, only_paths)
        if removed_unrequested:
            print("FastSim HLT: removed unrequested HLT menu paths: %s" %
                  _format_path_list(removed_unrequested))
    else:
        removed_unrequested = []
    _set_audit_vstring(process, "removedUnrequestedPaths", removed_unrequested)
    profiler.mark("prune unrequested paths")

    if enable_review_default:
        removed_review_default_paths = _remove_review_default_unsupported_paths(process, keep_paths)
        if removed_review_default_paths:
            print("FastSim HLT: review default pruned unsupported/experimental paths: %s" %
                  _format_path_list(removed_review_default_paths))
    else:
        removed_review_default_paths = []
    _set_audit_vstring(process, "removedReviewDefaultUnsupportedPaths", removed_review_default_paths)
    profiler.mark("prune review-default unsupported paths")

    if use_combined_pruner:
        combined_removed_paths = _remove_scheduled_paths_with_combined_pruner(process, keep_paths)
        removed_paths = combined_removed_paths["mc"]
        removed_mkfit_paths = combined_removed_paths["mkfit_string"]
        removed_mkfit_module_paths = combined_removed_paths["mkfit_module"]
        removed_any_mkfit_paths = combined_removed_paths["mkfit_any"]
        removed_tracking_paths = combined_removed_paths["tracking"]
        profiler.mark("combined path pruning")
    else:
        removed_paths = _remove_scheduled_paths(process, "MC_", keep_paths)
        profiler.mark("prune MC paths")

        removed_mkfit_paths = _remove_scheduled_paths_using(
            process,
            "hltIter0PFlowCkfTrackCandidatesMkFitSeeds",
            keep_paths,
        )
        profiler.mark("prune mkFit by dumped path text")

        removed_mkfit_module_paths = _remove_scheduled_paths_with_modules(
            process,
            (
                "hltDoubletRecoveryPFlowCkfTrackCandidatesMkFitSeeds",
                "hltIter0PFlowCkfTrackCandidatesMkFitSeeds",
            ),
            keep_paths,
        )
        profiler.mark("prune mkFit seed modules")

        removed_any_mkfit_paths = _remove_scheduled_paths_with_module_name_fragment(
            process,
            "MkFit",
            keep_paths,
        )
        profiler.mark("prune remaining MkFit modules")

        removed_tracking_paths = []
        for sequence_name in ("HLTTrackingForBeamSpot", "HLTPFScoutingTrackingSequence"):
            removed_tracking_paths.extend(_remove_scheduled_paths_using(process, sequence_name, keep_paths))
        profiler.mark("prune tracking sequences")

    if removed_paths:
        print("FastSim HLT: removed unsupported MC paths: %s" % _format_path_list(removed_paths))
    _set_audit_vstring(process, "removedMCPaths", removed_paths)
    if removed_mkfit_paths:
        print("FastSim HLT: removed paths using unsupported mkFit seeds: %s" % _format_path_list(removed_mkfit_paths))
    _set_audit_vstring(process, "removedMkFitStringPaths", removed_mkfit_paths)
    if removed_mkfit_module_paths:
        print("FastSim HLT: removed paths containing unsupported mkFit seed modules: %s" %
              _format_path_list(removed_mkfit_module_paths))
    _set_audit_vstring(process, "removedMkFitModulePaths", removed_mkfit_module_paths)
    if removed_any_mkfit_paths:
        print("FastSim HLT: removed paths containing unsupported MkFit modules: %s" %
              _format_path_list(removed_any_mkfit_paths))
    _set_audit_vstring(process, "removedMkFitAnyPaths", removed_any_mkfit_paths)
    if removed_tracking_paths:
        print("FastSim HLT: removed paths using unsupported tracking sequences: %s" %
              _format_path_list(removed_tracking_paths))
    _set_audit_vstring(process, "removedTrackingSequencePaths", removed_tracking_paths)

    if enable_muon_pixel_tracks_prototype:
        muon_pixel_tracks_prototype_target_paths = only_paths if only_paths else ()
        muon_pixel_tracks_prototype_paths = _install_fast_hlt_pixel_tracks(
            process,
            muon_pixel_tracks_prototype_target_paths,
            scheduled_only=True,
        )
        _set_audit_vstring(
            process,
            "muonPixelTracksPrototypePaths",
            muon_pixel_tracks_prototype_paths,
        )
    else:
        muon_pixel_tracks_prototype_paths = []
    profiler.mark("install scheduled muon pixel-track substitutions")
    profiler.mark("customizer total")

    return process


def customizeFastSimHLT(process):
    """Apply FastSim HLT customizations controlled by FASTSIM_HLT_* switches."""
    return _customizeFastSimHLT(process)


def customizeFastSimHLTGRunReviewDefault(process):
    """Apply the review-default FastSim HLT:GRun prototype."""
    return _customizeFastSimHLT(process, force_review_default=True)
