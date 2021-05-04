import FWCore.ParameterSet.Config as cms

cosmicsVetoSeeds = cms.EDProducer("TrajectorySeedFromMuonProducer",
                                muonCollectionTag = cms.InputTag("muons1stStep"),
                                trackCollectionTag = cms.InputTag("generalTracks"),
                                skipMatchedMuons = cms.bool(False)
                                )

from RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff import *
cosmicsVetoTrackCandidates = ckfTrackCandidatesP5.clone(
    src                       = "cosmicsVetoSeeds",
    doSeedingRegionRebuilding = False,
    RedundantSeedCleaner      = "none"
)
from RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff import *
cosmicsVetoTracksRaw = ctfWithMaterialTracksCosmics.clone(
    src = "cosmicsVetoTrackCandidates"
)
# need to clone FittingSmootherRKP5 if I want to change its parameters
# process.FittingSmootherRKP5.EstimateCut = cms.double(-1.0) # turn off the OutlierRejection

from RecoTracker.FinalTrackSelectors.cosmictrackSelector_cfi import *
cosmicsVetoTracks = cosmictrackSelector.clone(
    src = "cosmicsVetoTracksRaw"
)

from RecoMuon.MuonIdentification.muonCosmicCompatibility_cfi import *

cosmicsVeto = cms.EDProducer("CosmicsMuonIdProducer",
    MuonCosmicCompatibilityParameters,
    muonCollection = cms.InputTag("muons1stStep"),
    trackCollections = cms.VInputTag(cms.InputTag("generalTracks"), cms.InputTag("cosmicsVetoTracks")) 
)

cosmicsMuonIdTask = cms.Task(cosmicsVetoSeeds,cosmicsVetoTrackCandidates,cosmicsVetoTracksRaw,cosmicsVetoTracks,cosmicsVeto)
cosmicsMuonIdSequence = cms.Sequence(cosmicsMuonIdTask)
