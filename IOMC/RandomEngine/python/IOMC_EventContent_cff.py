import FWCore.ParameterSet.Config as cms

#Full Event content 
IOMCFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_randomEngineStateProducer_*_*')
)

