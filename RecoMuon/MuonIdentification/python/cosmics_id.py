import FWCore.ParameterSet.Config as cms

# -*-SH-*-
cosmicsVetoSeeds = cms.EDProducer("TrajectorySeedFromMuonProducer"
                                ,muonCollectionTag = cms.InputTag("muons1stStep")
                                ,trackCollectionTag = cms.InputTag("generalTracks")
                                # ,skipMatchedMuons = cms.bool(True)
                                ,skipMatchedMuons = cms.bool(False)
                                )

from RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff import *
cosmicsVetoTrackCandidates = copy.deepcopy(ckfTrackCandidatesP5)
cosmicsVetoTrackCandidates.src = cms.InputTag("cosmicsVetoSeeds")
cosmicsVetoTrackCandidates.doSeedingRegionRebuilding = False
cosmicsVetoTrackCandidates.RedundantSeedCleaner = "none"

from RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff import *
cosmicsVetoTracksRaw = copy.deepcopy(ctfWithMaterialTracksCosmics)
cosmicsVetoTracksRaw.src = cms.InputTag("cosmicsVetoTrackCandidates")
# need to clone FittingSmootherRKP5 if I want to change its parameters
# process.FittingSmootherRKP5.EstimateCut = cms.double(-1.0) # turn off the OutlierRejection

from RecoTracker.FinalTrackSelectors.cosmictrackSelector_cfi import *
cosmicsVetoTracks = cosmictrackSelector.clone(
    src = cms.InputTag("cosmicsVetoTracksRaw")
)

from RecoMuon.MuonIdentification.muonCosmicCompatibility_cfi import *

cosmicsVeto = cms.EDProducer("CosmicsMuonIdProducer"
    ,MuonCosmicCompatibilityParameters 
    ,muonCollection = cms.InputTag("muons1stStep")
    ,trackCollections = cms.VInputTag(cms.InputTag("generalTracks"), cms.InputTag("cosmicsVetoTracks")) 

    )

cosmicsMuonIdSequence = cms.Sequence(cosmicsVetoSeeds*cosmicsVetoTrackCandidates*cosmicsVetoTracksRaw*cosmicsVetoTracks*cosmicsVeto)


