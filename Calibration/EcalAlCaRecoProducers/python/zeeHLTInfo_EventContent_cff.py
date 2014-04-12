import FWCore.ParameterSet.Config as cms

# zeeMCInfo_EventContent.cff #########
HLTInfo = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmTriggerResults_*_*_*')
)

