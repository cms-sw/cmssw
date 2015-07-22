import FWCore.ParameterSet.Config as cms

#
# ctf tracks parameter-set entries for module
#
# SimpleTrackListMerger
#
# located in
#
# RecoTracker/FinalTrackSelectors
#
# 
# sequence dependency:
#
# - ctfWithMaterialTracks: include "RecoTracker/TrackProducer/data/CTFFinalFitWithMaterial.cff"
# - rsWithMaterialTracks: include "RecoTracker/TrackProducer/data/RSFinalFitWithMaterial.cff"
#
#
# cleans and merges ctf and rs Track lists and put new list back in Event

simpleTrackListMerger = cms.EDProducer("SimpleTrackListMerger",
    # minimum shared fraction to be called duplicate
    ShareFrac = cms.double(0.19),
    # best track chosen by chi2 modified by parameters below:
    FoundHitBonus = cms.double(5.0),
    LostHitPenalty = cms.double(10.0),
    # minimum pT in GeV/c
    MinPT = cms.double(0.05),
    # minimum difference in rechit position in cm
    # negative Epsilon uses sharedInput for comparison
    Epsilon = cms.double(-0.001),
    # maximum chisq/dof
    MaxNormalizedChisq = cms.double(1000.0),
    # minimum number of RecHits used in fit
    MinFound = cms.int32(3),
    # module laber of RS Tracks from KF with material propagator
    TrackProducer2 = cms.string(''),
    # module laber of CTF Tracks from KF with material propagator
    TrackProducer1 = cms.string(''),
    # set new quality for confirmed tracks
    promoteTrackQuality = cms.bool(False),
    allowFirstHitShare = cms.bool(True),
    newQuality = cms.string('confirmed'),
    copyExtras = cms.untracked.bool(False)
)


