# The following comments couldn't be translated into the new config version:

# L2 isolation producer and selector
import FWCore.ParameterSet.Config as cms

# keep essential info used to construct Tau HLT.
SingleTauMETHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2SingleTauMETJets*_*_*', 
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

