import FWCore.ParameterSet.Config as cms

# zeeMCInfo_EventContent.cff #########
MCInfo = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_zeeMCFilter_*_*')
)

