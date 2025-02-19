import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagMC_QCD_800_1000_cfi import *
btagMC_QCD_800_1000EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
btagMC_QCD_800_1000EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('btagMC_QCD_800_1000Path')
    )
)

