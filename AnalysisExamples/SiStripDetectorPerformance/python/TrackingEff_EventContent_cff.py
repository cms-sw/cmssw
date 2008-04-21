import FWCore.ParameterSet.Config as cms

TrackingEffEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep recoCandidatesOwned_genParticleCandidates_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoMuons_trackerMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep edmTriggerResults_*_*_*', 
        'keep recoHLTGlobalObject_*_*_*', 
        'keep recoHLTPathObject_*_*_*', 
        'keep recoHLTFilterObjectBase_*_*_*', 
        'keep recoHLTFilterObject_*_*_*', 
        'keep recoHLTFilterObjectWithRefs_*_*_*', 
        'keep triggerTriggerEvent_*_*_*', 
        'keep triggerTriggerEventWithRefs_*_*_*', 
        'keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*', 
        'keep *_l1extraParticles_*_*', 
        'keep *_l1extraParticleMap_*_*')
)
TrackingEffEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('TrackingEffPath')
    )
)

