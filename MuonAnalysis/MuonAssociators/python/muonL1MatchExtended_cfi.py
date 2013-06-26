import FWCore.ParameterSet.Config as cms

from MuonAnalysis.MuonAssociators.muonL1Match_cfi import *
from math import pi

muonL1MatchExtended = cms.EDProducer("L1MatcherExtended",
    muons   = cms.InputTag("muons"),
    l1extra = cms.InputTag("l1extraParticles"),
    segmentArbitration = cms.string("SegmentAndTrackArbitration"),
    csctfDigis = cms.InputTag("csctfDigis"),
    csctfLcts  = cms.InputTag("csctfDigis"),
    matcherGeom = cms.PSet(
        preselection = cms.string("gmtMuonCand.quality > 1"), # FIXME: maybe exclude CSC-only region?
        useTrack  = cms.string("tracker"),
        useState  = cms.string("atVertex"),
        maxDeltaR   = cms.double(1.5),             ## FIXME: to be tuned
        maxDeltaEta = cms.double(0.3),             ## FIXME: to be tuned
        l1PhiOffset = cms.double(1.25 * pi/180.),  
        useSimpleGeometry = cms.bool(True),
        fallbackToME1     = cms.bool(True),
    ) 
)

def addUserData(patMuonProducer, matcherLabel='muonL1MatchExtended', addExtraInfo=False):
    patMuonProducer.userData.userInts.src += [  cms.InputTag(matcherLabel) ]
    if addExtraInfo:
        for L in ("cscMode", "canPropagate", "l1q"):
             patMuonProducer.userData.userInts.src += [  cms.InputTag(matcherLabel,L) ]
        for L in ("deltaR", "deltaEta", "deltaPhi", "l1pt"):
             patMuonProducer.userData.userFloats.src += [  cms.InputTag(matcherLabel,L) ]

