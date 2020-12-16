import FWCore.ParameterSet.Config as cms

#from RecoHI.HiTracking.hiMultiTrackSelector_cfi import *

# Track selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiInitialStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src            = 'hiGlobalPrimTracks',
    useAnyMVA      = True,
    GBRForestLabel = 'HIMVASelectorIter4',
    GBRForestVars  = ['chi2perdofperlayer', 'dxyperdxyerror', 'dzperdzerror', 'nhits', 'nlayers', 'eta'],
    trackSelectors = cms.VPSet(
    	RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    	    name   = 'hiInitialStepLoose',
    	    useMVA = False
    	    ), #end of pset
    	RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    	    name          = 'hiInitialStepTight',
    	    preFilterName = 'hiInitialStepLoose',
    	    useMVA        = True,
    	    minMVA        = -0.77
    	    ),
    	RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    	    name          = 'hiInitialStep',
    	    preFilterName = 'hiInitialStepTight',
    	    useMVA        = True,
    	    minMVA        = -0.77
    	    ),
    ) #end of vpset
) #end of clone  
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiInitialStepSelector, useAnyMVA = False)
trackingPhase1.toModify(hiInitialStepSelector, trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
        name   = 'hiInitialStepLoose',
        useMVA = False
        ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
        name          = 'hiInitialStepTight',
        preFilterName = 'hiInitialStepLoose',
        useMVA        = False,
        minMVA        = -0.77
        ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
        name          = 'hiInitialStep',
        preFilterName = 'hiInitialStepTight',
        useMVA        = False,
        minMVA        = -0.77
        ),
    ) #end of vpset
)



# using the tracklist merger with one collection simply applies the quality flags
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiSelectedTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers     = ['hiGlobalPrimTracks'],
    hasSelector        = [1],
    selectedTrackQuals = ["hiInitialStepSelector:hiInitialStep"],
    copyExtras         = True,
    copyMVA            = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
)

#complete sequence
hiTracksWithQualityTask = cms.Task(hiInitialStepSelector
                                   #* hiSelectedTracks
)
