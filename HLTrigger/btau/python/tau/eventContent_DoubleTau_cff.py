import FWCore.ParameterSet.Config as cms

DoubleTauHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2DoubleTauJets_*_*', 
        'keep *_hltL2DoubleTauIsolation*_*_*', 
        'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 
        'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 
        'keep *_hltIsolatedL25PixelTau*_*_*')
)

