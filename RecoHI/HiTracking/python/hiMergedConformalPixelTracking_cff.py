import FWCore.ParameterSet.Config as cms

from RecoHI.HiTracking.HILowPtConformalPixelTracks_cfi import *
from RecoHI.HiTracking.hiMultiTrackSelector_cfi import *
from RecoTracker.FinalTrackSelectors.trackListMerger_cfi import *
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder

# Selector for quality pixel tracks with tapering high-pT cut

#loose
hiPixelOnlyStepLooseMTS = hiLooseMTS.clone(
    name = 'hiPixelOnlyTrkLoose',
    chi2n_no1Dmod_par = 25.0,
    d0_par2 = [9999.0, 0.0],              # d0E from tk.d0Error
    dz_par2 = [14.0, 0.0],
    max_relpterr = 9999.,
    min_nhits = 0,
    pixel_pTMinCut = cms.vdouble(0.0001,0.000,9999,1.0),
    pixel_pTMaxCut = cms.vdouble(10,5,25,2.5)
)

hiPixelOnlyStepTightMTS = hiPixelOnlyStepLooseMTS.clone(
    preFilterName ='hiPixelOnlyTrkLoose',
    chi2n_no1Dmod_par = 18.0,
    dz_par2 = [12.0, 0.0],
    pixel_pTMaxCut = [4,2,18,2.5],
    name = 'hiPixelOnlyTrkTight',
    qualityBit = 'tight',
    keepAllTracks = True
)

hiPixelOnlyStepHighpurityMTS= hiPixelOnlyStepTightMTS.clone(
    name = 'hiPixelOnlyTrkHighPurity',
    preFilterName ='hiPixelOnlyTrkTight',
    chi2n_no1Dmod_par = 12.,    
    dz_par2 = [10.0, 0.0],
    pixel_pTMaxCut = [2.4,1.6,12,2.5],
    qualityBit = 'highPurity' ## set to '' or comment out if you dont want to set the bit
)

hiPixelOnlyStepSelector = hiMultiTrackSelector.clone(
    applyPixelMergingCuts = cms.bool(True),
    src = 'hiConformalPixelTracks',
    trackSelectors= cms.VPSet(
        hiPixelOnlyStepLooseMTS,
        hiPixelOnlyStepTightMTS,
        hiPixelOnlyStepHighpurityMTS
    )
)


# selector for tapered full tracks

hiHighPtStepTruncMTS = hiLooseMTS.clone(
    name = 'hiHighPtTrkTrunc',
    chi2n_no1Dmod_par = 9999.0,
    d0_par2 = [9999.0, 0.0],              # d0E from tk.d0Error
    dz_par2 = [9999.0, 0.0],
    max_relpterr = 9999.,
    minHitsToBypassChecks = 9999,
    min_nhits = 12,
    pixel_pTMinCut = cms.vdouble(1.0,1.8,0.15,2.5),
    pixel_pTMaxCut = cms.vdouble(9998,9999,9999,1.0),
    qualityBit = ''
)

hiHighPtStepSelector = hiMultiTrackSelector.clone(
    applyPixelMergingCuts = cms.bool(True),
    src = 'hiGeneralTracks',
    trackSelectors= cms.VPSet(
        hiHighPtStepTruncMTS
    ) 
) 


hiGeneralAndPixelTracks = trackListMerger.clone(
    TrackProducers = ['hiConformalPixelTracks',
                      'hiGeneralTracks'],
    hasSelector = [1,1],
    selectedTrackQuals = ["hiPixelOnlyStepSelector:hiPixelOnlyTrkHighPurity",
                          "hiHighPtStepSelector:hiHighPtTrkTrunc"],
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
