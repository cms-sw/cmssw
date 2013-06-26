import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagMC_QCD_380_470_cfi import *
btagMC_QCD_380_470EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
btagMC_QCD_380_470EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('btagMC_QCD_380_470Path')
    )
)

