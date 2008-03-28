import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagMC_QCD_80_120_cfi import *
btagMC_QCD_80-120EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
btagMC_QCD_80-120EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('btagMC_QCD_80-120Path')
    )
)

