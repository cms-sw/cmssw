import FWCore.ParameterSet.Config as cms
from RecoTracker.IterativeTracking.MuonSeededStep_cff import *

###### Muon reconstruction module #####
from RecoTracker.FinalTrackSelectors.trackAlgoPriorityOrder_cfi import trackAlgoPriorityOrder
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiEarlyGeneralTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = [
	'hiGlobalPrimTracks',
        'hiDetachedTripletStepTracks',
        'hiLowPtTripletStepTracks',
        'hiPixelPairGlobalPrimTracks',
        'hiJetCoreRegionalStepTracks',
        'hiRegitMuInitialStepTracks',
        'hiRegitMuPixelPairStepTracks',
        'hiRegitMuMixedTripletStepTracks',
        'hiRegitMuPixelLessStepTracks',
        'hiRegitMuDetachedTripletStepTracks'
         ],
    hasSelector = [1,1,1,1,1,1,1,1,1,1],
    selectedTrackQuals = [
	"hiInitialStepSelector:hiInitialStep",
	"hiDetachedTripletStepSelector:hiDetachedTripletStep",
	"hiLowPtTripletStepSelector:hiLowPtTripletStep",
	"hiPixelPairStepSelector:hiPixelPairStep",
	"hiJetCoreRegionalStepSelector:hiJetCoreRegionalStep",
	"hiRegitMuInitialStepSelector:hiRegitMuInitialStepLoose",
	"hiRegitMuPixelPairStepSelector:hiRegitMuPixelPairStep",
	"hiRegitMuMixedTripletStepSelector:hiRegitMuMixedTripletStep",
	"hiRegitMuPixelLessStepSelector:hiRegitMuPixelLessStep",
	"hiRegitMuDetachedTripletStepSelector:hiRegitMuDetachedTripletStep"
     	 ],
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6,7,8,9), pQual=cms.bool(True)),  # should this be False?
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )

hiEarlyMuons = earlyMuons.clone(
    inputCollectionLabels = ["hiEarlyGeneralTracks", "standAloneMuons:UpdatedAtVtx"]
)

###### SEEDER MODELS ######
import RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi
import RecoTracker.SpecialSeedGenerators.inOutSeedsFromTrackerMuons_cfi
hiRegitMuonSeededSeedsOutIn = RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi.outInSeedsFromStandaloneMuons.clone(
    src = "hiEarlyMuons",
)
hiRegitMuonSeededSeedsInOut = RecoTracker.SpecialSeedGenerators.inOutSeedsFromTrackerMuons_cfi.inOutSeedsFromTrackerMuons.clone(
    src = "hiEarlyMuons",
)

hiRegitMuonSeededTrackCandidatesInOut = muonSeededTrackCandidatesInOut.clone(
    src = "hiRegitMuonSeededSeedsInOut"
)
hiRegitMuonSeededTrackCandidatesOutIn = muonSeededTrackCandidatesOutIn.clone(
    src = "hiRegitMuonSeededSeedsOutIn"
)

hiRegitMuonSeededTracksOutIn = muonSeededTracksOutIn.clone(
    src = "hiRegitMuonSeededTrackCandidatesOutIn",
    AlgorithmName = 'hiRegitMuMuonSeededStepOutIn'
)
hiRegitMuonSeededTracksInOut = muonSeededTracksInOut.clone(
    src = "hiRegitMuonSeededTrackCandidatesInOut",
    AlgorithmName = 'hiRegitMuMuonSeededStepInOut'
)

import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitMuonSeededTracksInOutSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
      src            = 'hiRegitMuonSeededTracksInOut',
      vertices       = "hiSelectedPixelVertex",
      useAnyMVA      = True,
      GBRForestLabel = 'HIMVASelectorIter7',
      GBRForestVars  = ['chi2perdofperlayer', 'nhits', 'nlayers', 'eta'],
      trackSelectors = cms.VPSet(
         RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name      = 'hiRegitMuonSeededTracksInOutLoose',
            min_nhits = 8
            ),
         RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name          = 'hiRegitMuonSeededTracksInOutTight',
            preFilterName = 'hiRegitMuonSeededTracksInOutLoose',
            min_nhits     = 8,
            useMVA        = True,
            minMVA        = -0.2
            ),
         RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name          = 'hiRegitMuonSeededTracksInOutHighPurity',
            preFilterName = 'hiRegitMuonSeededTracksInOutTight',
            min_nhits     = 8,
            useMVA        = True,
            minMVA        = -0.09
            ),
         ) #end of vpset
) #end of clone
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiRegitMuonSeededTracksInOutSelector, useAnyMVA = False)
trackingPhase1.toModify(hiRegitMuonSeededTracksInOutSelector, trackSelectors= cms.VPSet(
         RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name      = 'hiRegitMuonSeededTracksInOutLoose',
            min_nhits = 8
            ),
         RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name          = 'hiRegitMuonSeededTracksInOutTight',
            preFilterName = 'hiRegitMuonSeededTracksInOutLoose',
            min_nhits     = 8,
            useMVA        = False,
            minMVA        = -0.2
            ),
         RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name          = 'hiRegitMuonSeededTracksInOutHighPurity',
            preFilterName = 'hiRegitMuonSeededTracksInOutTight',
            min_nhits     = 8,
            useMVA        = False,
            minMVA        = -0.09
            ),
         ) #end of vpset
)

hiRegitMuonSeededTracksOutInSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
      src            = 'hiRegitMuonSeededTracksOutIn',
      vertices       = "hiSelectedPixelVertex",
      useAnyMVA      = True,
      GBRForestLabel = 'HIMVASelectorIter7',
      GBRForestVars  = ['chi2perdofperlayer', 'nhits', 'nlayers', 'eta'],
      trackSelectors = cms.VPSet(
         RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name      = 'hiRegitMuonSeededTracksOutInLoose',
            min_nhits = 8
            ),
         RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name          = 'hiRegitMuonSeededTracksOutInTight',
            preFilterName = 'hiRegitMuonSeededTracksOutInLoose',
            min_nhits     = 8,
            useMVA        = True,
            minMVA        = -0.2
            ),
         RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name          = 'hiRegitMuonSeededTracksOutInHighPurity',
            preFilterName = 'hiRegitMuonSeededTracksOutInTight',
            min_nhits     = 8,
            useMVA        = True,
            minMVA        = -0.09
            ),
         ) #end of vpset
) #end of clone
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiRegitMuonSeededTracksOutInSelector, useAnyMVA = False)
trackingPhase1.toModify(hiRegitMuonSeededTracksOutInSelector, trackSelectors= cms.VPSet(
         RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name      = 'hiRegitMuonSeededTracksOutInLoose',
            min_nhits = 8
            ),
         RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name          = 'hiRegitMuonSeededTracksOutInTight',
            preFilterName = 'hiRegitMuonSeededTracksOutInLoose',
            min_nhits     = 8,
            useMVA        = False,
            minMVA        = -0.2
            ),
         RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name          = 'hiRegitMuonSeededTracksOutInHighPurity',
            preFilterName = 'hiRegitMuonSeededTracksOutInTight',
            min_nhits     = 8,
            useMVA        = False,
            minMVA        = -0.09
            ),
         ) #end of vpset
)

hiRegitMuonSeededStepCoreTask = cms.Task(
      hiRegitMuonSeededSeedsInOut , hiRegitMuonSeededTrackCandidatesInOut , hiRegitMuonSeededTracksInOut ,
      hiRegitMuonSeededSeedsOutIn , hiRegitMuonSeededTrackCandidatesOutIn , hiRegitMuonSeededTracksOutIn 
      )
hiRegitMuonSeededStepExtraTask = cms.Task(
      hiRegitMuonSeededTracksInOutSelector ,
      hiRegitMuonSeededTracksOutInSelector
      )
hiRegitMuonSeededStepTask = cms.Task(
      hiEarlyGeneralTracks ,
      hiEarlyMuons ,
      hiRegitMuonSeededStepCoreTask ,
      hiRegitMuonSeededStepExtraTask 
      )
hiRegitMuonSeededStep = cms.Sequence(hiRegitMuonSeededStepTask)
