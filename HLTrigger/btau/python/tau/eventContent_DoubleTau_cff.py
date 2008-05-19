# The following comments couldn't be translated into the new config version:

# L2 isolation producer and selector
import FWCore.ParameterSet.Config as cms

# keep essential info used to construct Tau HLT.
DoubleTauHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2DoubleTauJets*_*_*', 
        'keep *_hltL2DoubleTauIsolation*_*_*', 
        'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 
        'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 
        'keep *_hltIsolatedL25PixelTau*_*_*')
)

