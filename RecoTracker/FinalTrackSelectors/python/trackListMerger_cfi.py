import FWCore.ParameterSet.Config as cms

#
# ctf tracks parameter-set entries for module
#
# TrackListMerger
#
# located in
#
# RecoTracker/FinalTrackSelectors
#
# 
# sequence dependency:
#
#
#
# cleans and merges ctf and rs Track lists and put new list back in Event

trackListMerger = cms.EDProducer("TrackListMerger",
    # minimum shared fraction to be called duplicate for tracks between collections
    ShareFrac = cms.double(0.19),
    # best track chosen by chi2 modified by parameters below:
    FoundHitBonus = cms.double(5.0),
    LostHitPenalty = cms.double(20.0),
    # minimum pT in GeV/c
    MinPT = cms.double(0.05),
    # minimum difference in rechit position in cm
    # negative Epsilon uses sharedInput for comparison
    Epsilon = cms.double(-0.001),
    # maximum chisq/dof
    MaxNormalizedChisq = cms.double(1000.0),
    # minimum number of RecHits used in fit
    MinFound = cms.int32(3),
    # always override these in the clone                             
    TrackProducers = cms.VInputTag(cms.InputTag(''),cms.InputTag('')),
    hasSelector = cms.vint32(0,0),
    # minimum shared fraction to be called duplicate
    indivShareFrac = cms.vdouble(1.0,1.0),
    selectedTrackQuals = cms.VInputTag(cms.InputTag(""),cms.InputTag("")),                             
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(False)),
                             cms.PSet( tLists=cms.vint32(2,3), pQual=cms.bool(True) ),
                             cms.PSet( tLists=cms.vint32(4,5), pQual=cms.bool(True) ),
                             cms.PSet( tLists=cms.vint32(2,3,4,5), pQual=cms.bool(True) ),
                             cms.PSet( tLists=cms.vint32(0,1,2,3,4,5), pQual=cms.bool(True) )
                             ),

    # set new quality for confirmed tracks for each merged pair and then for the final pair
    allowFirstHitShare = cms.bool(True),
    newQuality = cms.string('confirmed'),
    copyExtras = cms.untracked.bool(False),
    writeOnlyTrkQuals = cms.bool(False)

)


