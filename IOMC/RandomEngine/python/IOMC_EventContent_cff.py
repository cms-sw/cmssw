import FWCore.ParameterSet.Config as cms

#RAW content 
IOMCRAW = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_randomEngineStateProducer_*_*')
)

