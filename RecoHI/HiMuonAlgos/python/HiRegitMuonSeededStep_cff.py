import FWCore.ParameterSet.Config as cms
from RecoTracker.IterativeTracking.MuonSeededStep_cff import *

###### Muon reconstruction module #####
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiEarlyGeneralTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = (cms.InputTag('hiGlobalPrimTracks'),
                      cms.InputTag('hiDetachedTripletStepTracks'),
                      cms.InputTag('hiLowPtTripletStepTracks'),
                      cms.InputTag('hiPixelPairGlobalPrimTracks'),
                      cms.InputTag('hiJetCoreRegionalStepTracks'),
                      cms.InputTag('hiRegitMuInitialStepTracks'),
                      cms.InputTag('hiRegitMuPixelPairStepTracks'),
                      cms.InputTag('hiRegitMuMixedTripletStepTracks'),
                      cms.InputTag('hiRegitMuPixelLessStepTracks'),
                      cms.InputTag('hiRegitMuDetachedTripletStepTracks')
                     ),
    hasSelector=cms.vint32(1,1,1,1,1,1,1,1,1,1),
    selectedTrackQuals = cms.VInputTag(
    cms.InputTag("hiInitialStepSelector","hiInitialStep"),
    cms.InputTag("hiDetachedTripletStepSelector","hiDetachedTripletStep"),
    cms.InputTag("hiLowPtTripletStepSelector","hiLowPtTripletStep"),
    cms.InputTag("hiPixelPairStepSelector","hiPixelPairStep"),
    cms.InputTag("hiJetCoreRegionalStepSelector","hiJetCoreRegionalStep"),
    cms.InputTag("hiRegitMuInitialStepSelector","hiRegitMuInitialStepLoose"),
    cms.InputTag("hiRegitMuPixelPairStepSelector","hiRegitMuPixelPairStep"),
    cms.InputTag("hiRegitMuMixedTripletStepSelector","hiRegitMuMixedTripletStep"),
    cms.InputTag("hiRegitMuPixelLessStepSelector","hiRegitMuPixelLessStep"),
    cms.InputTag("hiRegitMuDetachedTripletStepSelector","hiRegitMuDetachedTripletStep")
    ),                    
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6,7,8,9), pQual=cms.bool(True)),  # should this be False?
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
    )

hiEarlyMuons = earlyMuons.clone(
      inputCollectionLabels = cms.VInputTag(cms.InputTag("hiEarlyGeneralTracks"),cms.InputTag("standAloneMuons","UpdatedAtVtx"))
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
      src = cms.InputTag("hiRegitMuonSeededSeedsInOut")
      )
hiRegitMuonSeededTrackCandidatesOutIn = muonSeededTrackCandidatesOutIn.clone(
      src = cms.InputTag("hiRegitMuonSeededSeedsOutIn")
      )

hiRegitMuonSeededTracksOutIn = muonSeededTracksOutIn.clone(
      src = cms.InputTag("hiRegitMuonSeededTrackCandidatesOutIn"),
      AlgorithmName = cms.string('hiRegitMuMuonSeededStepOutIn') 
      )
hiRegitMuonSeededTracksInOut = muonSeededTracksInOut.clone(
      src = cms.InputTag("hiRegitMuonSeededTrackCandidatesInOut"),
      AlgorithmName = cms.string('hiRegitMuMuonSeededStepInOut') 
      )

import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitMuonSeededTracksInOutSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
      src='hiRegitMuonSeededTracksInOut',
      vertices            = cms.InputTag("hiSelectedVertex"),
      useAnyMVA = cms.bool(True),
      GBRForestLabel = cms.string('HIMVASelectorIter7'),
      GBRForestVars = cms.vstring(['chi2perdofperlayer', 'nhits', 'nlayers', 'eta']),
      trackSelectors= cms.VPSet(
         RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'hiRegitMuonSeededTracksInOutLoose',
            min_nhits = cms.uint32(8)
            ),
         RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiRegitMuonSeededTracksInOutTight',
            preFilterName = 'hiRegitMuonSeededTracksInOutLoose',
            min_nhits = cms.uint32(8),
            useMVA = cms.bool(True),
            minMVA = cms.double(-0.2)
            ),
         RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiRegitMuonSeededTracksInOutHighPurity',
            preFilterName = 'hiRegitMuonSeededTracksInOutTight',
            min_nhits = cms.uint32(8),
            useMVA = cms.bool(True),
            minMVA = cms.double(-0.09)
            ),
         ) #end of vpset
      ) #end of clone

hiRegitMuonSeededTracksOutInSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
      src='hiRegitMuonSeededTracksOutIn',
      vertices            = cms.InputTag("hiSelectedVertex"),
      useAnyMVA = cms.bool(True),
      GBRForestLabel = cms.string('HIMVASelectorIter7'),
      GBRForestVars = cms.vstring(['chi2perdofperlayer', 'nhits', 'nlayers', 'eta']),
      trackSelectors= cms.VPSet(
         RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'hiRegitMuonSeededTracksOutInLoose',
            min_nhits = cms.uint32(8)
            ),
         RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiRegitMuonSeededTracksOutInTight',
            preFilterName = 'hiRegitMuonSeededTracksOutInLoose',
            min_nhits = cms.uint32(8),
            useMVA = cms.bool(True),
            minMVA = cms.double(-0.2)
            ),
         RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiRegitMuonSeededTracksOutInHighPurity',
            preFilterName = 'hiRegitMuonSeededTracksOutInTight',
            min_nhits = cms.uint32(8),
            useMVA = cms.bool(True),
            minMVA = cms.double(-0.09)
            ),
         ) #end of vpset
      ) #end of clone

hiRegitMuonSeededStepCore = cms.Sequence(
      hiRegitMuonSeededSeedsInOut + hiRegitMuonSeededTrackCandidatesInOut + hiRegitMuonSeededTracksInOut +
      hiRegitMuonSeededSeedsOutIn + hiRegitMuonSeededTrackCandidatesOutIn + hiRegitMuonSeededTracksOutIn 
      )
hiRegitMuonSeededStepExtra = cms.Sequence(
      hiRegitMuonSeededTracksInOutSelector +
      hiRegitMuonSeededTracksOutInSelector
      )

hiRegitMuonSeededStep = cms.Sequence(
      hiEarlyGeneralTracks +
      hiEarlyMuons +
      hiRegitMuonSeededStepCore +
      hiRegitMuonSeededStepExtra 
      )
