import FWCore.ParameterSet.Config as cms

SingleTauHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2SingleTauJets_*_*', 
        'keep *_hltL2SingleTauIsolation*_*_*', 
        'keep *_hltMet*_*_*', 
        'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 
        'keep *_hltAssociatorL25SingleTau*_*_*', 
        'keep *_hltConeIsolationL25SingleTau*_*_*', 
        'keep *_hltIsolatedL25SingleTau*_*_*', 
        'keep *_hltCtfWithMaterialTracksL3SingleTau*_*_*', 
        'keep *_hltAssociatorL3SingleTau*_*_*', 
        'keep *_hltConeIsolationL3SingleTau*_*_*', 
        'keep *_hltIsolatedL3SingleTau*_*_*')
)

