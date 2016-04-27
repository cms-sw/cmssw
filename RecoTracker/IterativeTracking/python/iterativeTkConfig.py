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

def nonDefaultEras():
    return _nonDefaultEras

def createEarlySequence(era, modDict):
    postfix = "_"+era if era != _defaultEra else era
    _seq = cms.Sequence()
    for it in globals()["_iterations"+postfix]:
        _seq += modDict[it]
    return _seq
