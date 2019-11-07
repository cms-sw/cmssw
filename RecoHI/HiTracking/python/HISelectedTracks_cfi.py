import FWCore.ParameterSet.Config as cms

#from RecoHI.HiTracking.hiMultiTrackSelector_cfi import *

# Track selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiInitialStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiGlobalPrimTracks',
    useAnyMVA = cms.bool(True),
    GBRForestLabel = cms.string('HIMVASelectorIter4'),
    GBRForestVars = cms.vstring(['chi2perdofperlayer', 'dxyperdxyerror', 'dzperdzerror', 'nhits', 'nlayers', 'eta']),
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiInitialStepLoose',
    useMVA = cms.bool(False)
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiInitialStepTight',
    preFilterName = 'hiInitialStepLoose',
    useMVA = cms.bool(True),
    minMVA = cms.double(-0.77)
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiInitialStep',
    preFilterName = 'hiInitialStepTight',
    useMVA = cms.bool(True),
    minMVA = cms.double(-0.77)
    ),
    ) #end of vpset
    ) #end of clone  
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiInitialStepSelector, useAnyMVA = cms.bool(False))
trackingPhase1.toModify(hiInitialStepSelector, trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiInitialStepLoose',
    useMVA = cms.bool(False)
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiInitialStepTight',
    preFilterName = 'hiInitialStepLoose',
    useMVA = cms.bool(False),
    minMVA = cms.double(-0.77)
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiInitialStep',
    preFilterName = 'hiInitialStepTight',
    useMVA = cms.bool(False),
    minMVA = cms.double(-0.77)
    ),
    ) #end of vpset
)



# using the tracklist merger with one collection simply applies the quality flags
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiSelectedTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('hiGlobalPrimTracks')),
    hasSelector=cms.vint32(1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hiInitialStepSelector","hiInitialStep")),
    copyExtras = True,
    copyMVA = cms.bool(True),
    makeReKeyedSeeds = cms.untracked.bool(False)
    )

#complete sequence
hiTracksWithQualityTask = cms.Task(hiInitialStepSelector
                                   #* hiSelectedTracks
)
