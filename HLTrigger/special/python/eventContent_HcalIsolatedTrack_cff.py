import FWCore.ParameterSet.Config as cms

# Full Event content
HLTHcalIsolatedTrackFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ecalRecHit_*_*', 'keep *_ecalIsolPartProd_*_*', 'keep *_pixelTracks_*_*', 'keep *_isolPixelTrackProd_*_*')
)
# RECO content
HLTHcalIsolatedTrackRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ecalRecHit_*_*', 'keep *_ecalIsolPartProd_*_*', 'keep *_pixelTracks_*_*', 'keep *_isolPixelTrackProd_*_*')
)
# AOD content
HLTHcalIsolatedTrackAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("isolPixelTrackProd")),
    triggerFilters = cms.VInputTag(cms.InputTag("isolPixelTrackFilter")),
    outputCommands = cms.untracked.vstring()
)

