"""
Helpers for P2GT menus written in the structured form, i.e. with conditions/collections
that reference shared object definitions and use offline thresholds:

    SingleTkMuon22 = l1tGTSingleObjectCond.clone(
        object = cms.PSet(refToPSet_ = cms.string("l1tGTtkMuonVLoose")),
        offlineMinPt = cms.double(22),
    )

`flattenGTMenu(process)` rewrites all P2GT conditions of a process in place into the flat
form (object parameters inlined, online thresholds computed), exactly what the emulator
sees after L1GTObjectConfig.h resolved the configuration. Tools that inspect the python
configuration directly (e.g. the VHDL writer) should call it after loading the menu.
"""

import copy

import FWCore.ParameterSet.Config as cms

try:
    # same C++ implementation as used by the emulator
    from libL1TriggerPhase2L1GT import offlineToOnlinePt as _offlineToOnlinePt
except ImportError:
    _offlineToOnlinePt = None

GT_CONDITION_COLLECTIONS = {
    "L1GTSingleObjectCond": (),
    "L1GTDoubleObjectCond": ("collection1", "collection2"),
    "L1GTTripleObjectCond": ("collection1", "collection2", "collection3"),
    "L1GTQuadObjectCond": ("collection1", "collection2", "collection3", "collection4"),
}

STRUCTURE_PARAMETERS = ("object", "ptScaling", "offlineMinPt", "offlineMinScalarSumPt")


def offline_to_online_pt(offline, offset, slope, min_online=0.):
    """online = max(min_online, round((offline - offset) / slope, 1)), identical to l1t::offlineToOnlinePt"""
    if _offlineToOnlinePt is not None:
        return _offlineToOnlinePt(offline, offset, slope, min_online)
    return float(max(min_online, round((offline - offset) / slope, 1)))


def _deref(pset, process):
    seen = set()
    while pset.isRef_():
        name = pset.refToPSet_.value()
        if name in seen:
            raise RuntimeError("Circular refToPSet_ '%s'" % name)
        seen.add(name)
        if not hasattr(process, name):
            raise RuntimeError("refToPSet_ '%s' is not a top-level PSet of the process" % name)
        pset = getattr(process, name)
    return pset


def _online_thresholds(offline, scaling):
    offsets = scaling.regionsOffset.value()
    slopes = scaling.regionsSlope.value()
    min_online = scaling.minOnlinePt.value() if hasattr(scaling, "minOnlinePt") else 0.
    if len(offsets) == 0 or len(offsets) != len(slopes):
        raise RuntimeError("'ptScaling' needs 'regionsOffset' and 'regionsSlope' of equal, non-zero length")
    return [offline_to_online_pt(offline, o, s, min_online) for o, s in zip(offsets, slopes)]


def resolve_collection_parameters(params, process, where=""):
    """Flat parameters (name -> cms parameter) of a structured collection configuration.

    Mirrors l1t::resolveCollectionConfig: collection-level parameters override the object ones,
    offlineMinPt/offlineMinScalarSumPt are converted with 'ptScaling' into
    regionsMinPt (or minPt for a single region) / minScalarSumPt.
    """
    merged = {}
    if "object" in params:
        obj = _deref(params["object"], process)
        for name, value in obj.parameters_().items():
            merged[name] = copy.deepcopy(value)
    for name, value in params.items():
        if name != "object":
            merged[name] = copy.deepcopy(value)

    offline_pt = merged.get("offlineMinPt")
    offline_sum = merged.get("offlineMinScalarSumPt")
    if offline_pt is not None or offline_sum is not None:
        if "ptScaling" not in merged:
            raise RuntimeError("%s: offline threshold given but no 'ptScaling' defined" % where)
        scaling = _deref(merged["ptScaling"], process)

        if offline_pt is not None:
            thresholds = _online_thresholds(offline_pt.value(), scaling)
            target = "regionsMinPt" if len(thresholds) > 1 else "minPt"
            if target in merged:
                raise RuntimeError("%s: 'offlineMinPt' and '%s' are mutually exclusive" % (where, target))
            if len(thresholds) > 1:
                n_regions = len(merged["regionsAbsEtaLowerBounds"]) if "regionsAbsEtaLowerBounds" in merged else 0
                if n_regions != len(thresholds):
                    raise RuntimeError("%s: 'ptScaling' has %d regions, 'regionsAbsEtaLowerBounds' %d"
                                       % (where, len(thresholds), n_regions))
                merged[target] = cms.vdouble(*thresholds)
            else:
                merged[target] = cms.double(thresholds[0])

        if offline_sum is not None:
            thresholds = _online_thresholds(offline_sum.value(), scaling)
            if len(thresholds) != 1:
                raise RuntimeError("%s: 'offlineMinScalarSumPt' requires a single-region 'ptScaling'" % where)
            if "minScalarSumPt" in merged:
                raise RuntimeError("%s: 'offlineMinScalarSumPt' and 'minScalarSumPt' are mutually exclusive" % where)
            merged["minScalarSumPt"] = cms.double(thresholds[0])

    for name in STRUCTURE_PARAMETERS:
        merged.pop(name, None)
    return merged


def _set_parameters(target, params):
    for name in list(target.parameterNames_()):
        if name not in params:
            delattr(target, name)
    for name, value in params.items():
        setattr(target, name, value)


def is_gt_condition(module):
    return isinstance(module, cms.EDFilter) and module.type_() in GT_CONDITION_COLLECTIONS


def flatten_condition(module, process, label=""):
    collections = GT_CONDITION_COLLECTIONS[module.type_()]
    if not collections:
        _set_parameters(module, resolve_collection_parameters(module.parameters_(), process, label))
        return
    for col in collections:
        if hasattr(module, col):
            flat = resolve_collection_parameters(getattr(module, col).parameters_(), process, "%s.%s" % (label, col))
            setattr(module, col, cms.PSet(**flat))


def flattenGTMenu(process):
    """Rewrite all P2GT conditions of `process` in place into the flat form. Returns the process."""
    for label, module in process.filters_().items():
        if is_gt_condition(module):
            flatten_condition(module, process, label)

    # guard against conditions that were not converted
    for label, module in process.filters_().items():
        if not is_gt_condition(module):
            continue
        psets = [module] + [getattr(module, c) for c in GT_CONDITION_COLLECTIONS[module.type_()] if hasattr(module, c)]
        for pset in psets:
            left = [name for name in STRUCTURE_PARAMETERS if hasattr(pset, name)]
            if left:
                raise RuntimeError("Condition '%s' still has %s after flattening" % (label, left))
    return process
