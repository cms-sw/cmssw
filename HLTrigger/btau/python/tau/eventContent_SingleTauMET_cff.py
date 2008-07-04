import FWCore.ParameterSet.Config as cms

SingleTauMETHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2SingleTauMETJets_*_*', 
        'keep *_hltL2SingleTauMETIsolation*_*_*', 
        'keep *_hltMet*_*_*', 
        'keep *_hltAssociatorL25SingleTauMET*_*_*', 
        'keep *_hltConeIsolationL25SingleTauMET*_*_*', 
        'keep *_hltIsolatedL25SingleTauMET*_*_*', 
        'keep *_hltCtfWithMaterialTracksL3SingleTauMET*_*_*', 
        'keep *_hltAssociatorL3SingleTauMET*_*_*', 
        'keep *_hltConeIsolationL3SingleTauMET*_*_*', 
        'keep *_hltIsolatedL3SingleTauMET*_*_*')
)

