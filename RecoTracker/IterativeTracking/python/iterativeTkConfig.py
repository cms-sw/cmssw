# This file is the authoritative source of the order of tracking
# iterations. It is used in RECO, DQM, and VALIDATION. Note that here
# InitialStepPreSplitting is not counted as an iteration.
import FWCore.ParameterSet.Config as cms

_defaultEraName = ""
_nonDefaultEraNames = ["trackingLowPU", "trackingPhase1", "trackingPhase2PU140"]

# name, postfix, era
_defaultEra = (_defaultEraName, "", None)
_nonDefaultEras = [
    (_name, "_"+_name, getattr(__import__('Configuration.Eras.Modifier_'+_name+'_cff',globals(),locals(),[_name],0),_name)) \
    for _name in _nonDefaultEraNames
]

_allEras = [_defaultEra] + _nonDefaultEras


_iterations = [
    "InitialStep",
    "DetachedTripletStep",
    "LowPtTripletStep",
    "PixelPairStep",
    "MixedTripletStep",
    "PixelLessStep",
    "TobTecStep",
    "JetCoreRegionalStep",
]
_iterations_trackingLowPU = [
    "InitialStep",
    "LowPtTripletStep",
    "PixelPairStep",
    "DetachedTripletStep",
    "MixedTripletStep",
    "PixelLessStep",
    "TobTecStep",
]
_iterations_trackingPhase1 = [
    "InitialStep",
    "LowPtQuadStep",
    "HighPtTripletStep",
    "LowPtTripletStep",
    "DetachedQuadStep",
    "DetachedTripletStep",
    "PixelPairStep",
    "MixedTripletStep",
    "PixelLessStep",
    "TobTecStep",
]

from Configuration.ProcessModifiers.displacedTracking_cff import displacedTracking
displacedTracking.toModify(_iterations_trackingPhase1, func=lambda x: x.append('DisplacedGeneralStep'))

_iterations_trackingPhase1.append('JetCoreRegionalStep')

_iterations_trackingPhase2PU140 = [
    "InitialStep",
    "HighPtTripletStep",
    "LowPtQuadStep",
    "LowPtTripletStep",
    "DetachedQuadStep",
    "PixelPairStep",
]
from Configuration.ProcessModifiers.vectorHits_cff import vectorHits
vectorHits.toModify(_iterations_trackingPhase2PU140, func=lambda x: x.append('PixelLessStep'))
_iterations_muonSeeded = [
    "MuonSeededStepInOut",
    "MuonSeededStepOutIn",
]
_iterations_muonSeeded_trackingPhase1 = [
    "MuonSeededStepInOut",
    "MuonSeededStepOutIn",
]
#Phase2
_iterations_muonSeeded_trackingPhase2PU140 = [
    "MuonSeededStepInOut",
    "MuonSeededStepOutIn",
]
_multipleSeedProducers = {
    "MixedTripletStep": ["A", "B"],
    "TobTecStep": ["Pair", "Tripl"],
}
_multipleSeedProducers_trackingLowPU = {
    "MixedTripletStep": ["A", "B"],
}
_multipleSeedProducers_trackingPhase1 = {
    "PixelPairStep": ["A", "B"],
    "MixedTripletStep": ["A", "B"],
    "TobTecStep": ["Pair", "Tripl"],
}
from Configuration.ProcessModifiers.seedingDeepCore_cff import seedingDeepCore
seedingDeepCore.toModify(_multipleSeedProducers_trackingPhase1, func=lambda x: x.update({"JetCoreRegionalStep": ["Barrel","Endcap"]}))


_multipleSeedProducers_trackingPhase2PU140 = {}
_oldStyleHasSelector = set([
    "InitialStep",
    "HighPtTripletStep",
    "LowPtQuadStep",
    "LowPtTripletStep",
    "PixelPairStep",
    "PixelLessStep",
    "TobTecStep",
])

from Configuration.ProcessModifiers.displacedRegionalTracking_cff import displacedRegionalTracking
displacedRegionalTracking.toModify(_iterations_muonSeeded_trackingPhase1, func=lambda x: x.append('DisplacedRegionalStep'))
displacedRegionalTracking.toModify(_multipleSeedProducers_trackingPhase1, func=lambda x: x.update({'DisplacedRegionalStep': ['Pair', 'Tripl']}))

from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import trackClusterRemover as _trackClusterRemover
_trackClusterRemoverBase = _trackClusterRemover.clone(
    maxChi2                                  = 9.0,
    pixelClusters                            = "siPixelClusters",
    stripClusters                            = "siStripClusters",
    TrackQuality                             = 'highPurity',
    minNumberOfLayersWithMeasBeforeFiltering = 0,
)

#Phase2 : configuring the phase2 track Cluster Remover
from RecoLocalTracker.SubCollectionProducers.phase2trackClusterRemover_cfi import phase2trackClusterRemover as _phase2trackClusterRemover
_trackClusterRemoverBase_trackingPhase2PU140 = _phase2trackClusterRemover.clone(
    maxChi2                                  = 9.0,
    phase2pixelClusters                      = "siPixelClusters",
    phase2OTClusters                         = "siPhase2Clusters",
    TrackQuality                             = 'highPurity',
    minNumberOfLayersWithMeasBeforeFiltering = 0,
)

def _modulePrefix(iteration):
    return iteration[0].lower()+iteration[1:]

def _clusterRemover(iteration):
    return _modulePrefix(iteration)+"Clusters"

def _tracks(iteration):
    return _modulePrefix(iteration)+"Tracks"

def _classifier(iteration, oldStyle=False, oldStyleQualityMasks=False):
    pre = _modulePrefix(iteration)
    if oldStyle:
        if iteration in _oldStyleHasSelector:
            return pre+"Selector:" + ("QualityMasks" if oldStyleQualityMasks else pre)
        else:
            return pre
    else:
        return pre+":QualityMasks"

def allEras():
    return _allEras

def nonDefaultEras():
    return _nonDefaultEras

def createEarlyTask(eraName, postfix, modDict):
    task = cms.Task()
    for it in globals()["_iterations"+postfix]:
        task.add(modDict[it+'Task'])
    return task

def iterationAlgos(postfix, includeSequenceName=False):
    muonVariable = "_iterations_muonSeeded"+postfix
    iterations = globals()["_iterations"+postfix] + globals().get(muonVariable, _iterations_muonSeeded)

    if includeSequenceName:
        return [(_modulePrefix(i), i) for i in iterations]
    else:
        return [_modulePrefix(i) for i in iterations]

def _seedOrTrackProducers(postfix, typ):
    ret = []
    iters = globals()["_iterations"+postfix]
    if typ == "Seeds":
        multipleSeedProducers = globals()["_multipleSeedProducers"+postfix]
    else:
        multipleSeedProducers = None
    for i in iters:
        seeder = _modulePrefix(i)+typ
        if multipleSeedProducers is not None and i in multipleSeedProducers:
            ret.extend([seeder+m for m in multipleSeedProducers[i]])
        else:
            ret.append(seeder)

    for i in globals().get("_iterations_muonSeeded"+postfix, _iterations_muonSeeded):
        if _modulePrefix(i).endswith("Step"):
            ret.append(_modulePrefix(i)+typ)
        else:
            ret.append(_modulePrefix(i).replace("Step", typ))

    return ret

def seedProducers(postfix):
    return _seedOrTrackProducers(postfix, "Seeds")

def trackProducers(postfix):
    return _seedOrTrackProducers(postfix, "Tracks")

def clusterRemoverForIter(iteration, eraName="", postfix="", module=None):
    if module is None:
        module = _trackClusterRemoverBase.clone()
    if eraName == "trackingPhase2PU140":
        module = globals().get("_trackClusterRemoverBase"+postfix, _trackClusterRemoverBase)

    iters = globals()["_iterations"+postfix]
    try:
        # DisplacedRegionalStep is a special case because it comes after the
        # usual muon-seeded steps
        ind = iters.index(iteration) if iteration != "DisplacedRegionalStep" else len(iters)
    except ValueError:
        # if the iteration is not active in era, just return the same
        return module

    if ind == 0:
        raise Exception("Iteration %s is the first iteration in era %s, asking cluster remover configuration does not make sense" % (iteration, eraName))
    prevIter = iters[ind-1]
    # JetCoreRegionalStep uses all clusters, so if that is the previous
    # iteration, use the one before that for cluster removal
    if prevIter == "JetCoreRegionalStep":
        prevIter = iters[ind-2]

    customize = dict(
        trajectories          = _tracks(prevIter),
        oldClusterRemovalInfo = _clusterRemover(prevIter) if ind >= 2 else "", # 1st iteration does not have cluster remover
    )
    if eraName in ["trackingPhase2PU140"]:
        customize["overrideTrkQuals"] = _classifier(prevIter, oldStyle=True) # old-style selector
    elif eraName == "trackingLowPU":
        customize["overrideTrkQuals"] = _classifier(prevIter, oldStyle=True, oldStyleQualityMasks=True) # old-style selector with 'QualityMasks' instance label
    else:
        customize["trackClassifier"] = _classifier(prevIter)

    return module.clone(**customize)
