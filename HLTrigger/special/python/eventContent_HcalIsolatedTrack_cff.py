import FWCore.ParameterSet.Config as cms

# Full Event content
HLTHcalIsolatedTrackFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltSubdetFED_*_*', 
        'keep *_hltEcalRegFED_*_*', 
        'keep *_hltSiStripRegFED_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltIsolPixelTrackProd_*_*', 
        'keep *_hltL1sIsoTrack_*_*', 
        'keep *_hltGtDigis_*_*', 
        'keep l1extraL1JetParticles_hltL1extraParticles_*_*')
)
# RECO content
HLTHcalIsolatedTrackRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltSubdetFED_*_*', 
        'keep *_hltEcalRegFED_*_*', 
        'keep *_hltSiStripRegFED_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltIsolPixelTrackProd_*_*', 
        'keep *_hltL1sIsoTrack_*_*', 
        'keep *_hltGtDigis_*_*', 
        'keep l1extraL1JetParticles_hltL1extraParticles_*_*')
)
# AOD content
HLTHcalIsolatedTrackAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltIsolPixelTrackProd")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltIsolPixelTrackFilter")),
    outputCommands = cms.untracked.vstring()
)

