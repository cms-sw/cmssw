# The following comments couldn't be translated into the new config version:

# L2 isolation producer and selector     
import FWCore.ParameterSet.Config as cms

# keep essential info used to construct Tau HLT.
SingleTauHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2SingleTauJets*_*_*', 
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

