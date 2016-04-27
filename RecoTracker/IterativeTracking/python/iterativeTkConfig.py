# This file is the authoritative source of the order of tracking
# iterations. It is used in RECO, DQM, and VALIDATION. Note that here
# InitialStepPreSplitting is not counted as an iteration.
import FWCore.ParameterSet.Config as cms

_defaultEra = ""
_nonDefaultEras = ["trackingLowPU", "trackingPhase1", "trackingPhase1PU70"]
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
    "HighPtTripletStep",
    "DetachedQuadStep",
    #"DetachedTripletStep", # FIXME: dropped for time being, but it may be enabled on the course of further tuning
    "LowPtQuadStep",
    "LowPtTripletStep",
    "MixedTripletStep",
    "PixelLessStep",
    "TobTecStep",
    "JetCoreRegionalStep",
]
_iterations_trackingPhase1PU70 = [
    "InitialStep",
    "HighPtTripletStep",
    "LowPtQuadStep",
    "LowPtTripletStep",
    "DetachedQuadStep",
    "MixedTripletStep",
    "PixelPairStep",
    "TobTecStep",
]
_oldStyleHasSelector = set([
    "InitialStep",
    "HighPtTripletStep",
    "LowPtQuadStep",
    "LowPtTripletStep",
    "PixelPairStep",
    "PixelLessStep",
    "TobTecStep",
])

from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import trackClusterRemover as _trackClusterRemover
_trackClusterRemoverBase = _trackClusterRemover.clone(
    maxChi2                                  = 9.0,
    pixelClusters                            = "siPixelClusters",
    stripClusters                            = "siStripClusters",
    TrackQuality                             = 'highPurity',
    minNumberOfLayersWithMeasBeforeFiltering = 0,
)

def _postfix(era):
    return "_"+era if era != _defaultEra else era

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

def nonDefaultEras():
    return _nonDefaultEras

def createEarlySequence(era, modDict):
    postfix = _postfix(era)
    seq = cms.Sequence()
    for it in globals()["_iterations"+postfix]:
        seq += modDict[it]
    return seq

def clusterRemoverForIter(iteration, era="", module=None):
    if module is None:
        module = _trackClusterRemoverBase.clone()

    postfix = _postfix(era)
    iters = globals()["_iterations"+postfix]
    try:
        ind = iters.index(iteration)
    except ValueError:
        # if the iteration is not active in era, just return the same
        return module

    if ind == 0:
        raise Exception("Iteration %s is the first iteration in era %s, asking cluster remover configuration does not make sense" % (iteration, era))
    prevIter = iters[ind-1]

    customize = dict(
        trajectories          = _tracks(prevIter),
        oldClusterRemovalInfo = _clusterRemover(prevIter) if ind >= 2 else "", # 1st iteration does not have cluster remover
    )
    if era == "trackingPhase1PU70":
        customize["overrideTrkQuals"] = _classifier(prevIter, oldStyle=True) # old-style selector
    elif era == "trackingLowPU":
        customize["overrideTrkQuals"] = _classifier(prevIter, oldStyle=True, oldStyleQualityMasks=True) # old-style selector with 'QualityMasks' instance label
    else:
        customize["trackClassifier"] = _classifier(prevIter)

    return module.clone(**customize)
