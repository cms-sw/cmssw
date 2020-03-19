import FWCore.ParameterSet.Config as cms

from RecoHI.HiTracking.HILowPtConformalPixelTracks_cfi import *
from RecoHI.HiTracking.hiMultiTrackSelector_cfi import *
from RecoTracker.FinalTrackSelectors.trackListMerger_cfi import *
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder

# Selector for quality pixel tracks with tapering high-pT cut

#loose
hiPixelOnlyStepLooseMTS = hiLooseMTS.clone(
    name= cms.string('hiPixelOnlyTrkLoose'),
    chi2n_no1Dmod_par = cms.double(25.0),
    d0_par2 = cms.vdouble(9999.0, 0.0),              # d0E from tk.d0Error
    dz_par2 = cms.vdouble(14.0, 0.0), 
    max_relpterr = cms.double(9999.),
    min_nhits = cms.uint32(0),
    pixel_pTMinCut = cms.vdouble(0.0001,0.000,9999,1.0),
    pixel_pTMaxCut = cms.vdouble(10,5,25,2.5)
)

hiPixelOnlyStepTightMTS=hiPixelOnlyStepLooseMTS.clone(
    preFilterName='hiPixelOnlyTrkLoose',
    chi2n_no1Dmod_par = cms.double(18.0),
    dz_par2 = cms.vdouble(12.0, 0.0),
    pixel_pTMaxCut = cms.vdouble(4,2,18,2.5),
    name= cms.string('hiPixelOnlyTrkTight'),
    qualityBit = cms.string('tight'),
    keepAllTracks= cms.bool(True)
    )

hiPixelOnlyStepHighpurityMTS= hiPixelOnlyStepTightMTS.clone(
    name= cms.string('hiPixelOnlyTrkHighPurity'),
    preFilterName='hiPixelOnlyTrkTight',
    chi2n_no1Dmod_par = cms.double(12.),    
    dz_par2 = cms.vdouble(10.0, 0.0),
    pixel_pTMaxCut = cms.vdouble(2.4,1.6,12,2.5),
    qualityBit = cms.string('highPurity') ## set to '' or comment out if you dont want to set the bit
    )

hiPixelOnlyStepSelector = hiMultiTrackSelector.clone(
    applyPixelMergingCuts = cms.bool(True),
    src='hiConformalPixelTracks',
    trackSelectors= cms.VPSet(
        hiPixelOnlyStepLooseMTS,
        hiPixelOnlyStepTightMTS,
        hiPixelOnlyStepHighpurityMTS
    )
    )


# selector for tapered full tracks

hiHighPtStepTruncMTS = hiLooseMTS.clone(
    name= cms.string('hiHighPtTrkTrunc'),
    chi2n_no1Dmod_par = cms.double(9999.0),
    d0_par2 = cms.vdouble(9999.0, 0.0),              # d0E from tk.d0Error
    dz_par2 = cms.vdouble(9999.0, 0.0),
    max_relpterr = cms.double(9999.),
    minHitsToBypassChecks = cms.uint32(9999),
    min_nhits = cms.uint32(12),
    pixel_pTMinCut = cms.vdouble(1.0,1.8,0.15,2.5),
    pixel_pTMaxCut = cms.vdouble(9998,9999,9999,1.0),
    qualityBit = cms.string('')
)

hiHighPtStepSelector = hiMultiTrackSelector.clone(
    applyPixelMergingCuts = cms.bool(True),
    src='hiGeneralTracks',
    trackSelectors= cms.VPSet(
        hiHighPtStepTruncMTS
    ) 
    ) 


hiGeneralAndPixelTracks = trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('hiConformalPixelTracks'),
                          cms.InputTag('hiGeneralTracks')
                     ),
    hasSelector=cms.vint32(1,1),
    selectedTrackQuals = cms.VInputTag(
    cms.InputTag("hiPixelOnlyStepSelector","hiPixelOnlyTrkHighPurity"),
    cms.InputTag("hiHighPtStepSelector","hiHighPtTrkTrunc")
    ),                    
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(False)), 
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )

hiMergedConformalPixelTrackingTask = cms.Task(
    hiConformalPixelTracksTask
    ,hiPixelOnlyStepSelector
    ,hiHighPtStepSelector
    ,hiGeneralAndPixelTracks
    )
