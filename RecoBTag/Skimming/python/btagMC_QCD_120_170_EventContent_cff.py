import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagMC_QCD_120_170_cfi import *
btagMC_QCD_120-170EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
btagMC_QCD_120-170EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('btagMC_QCD_120-170Path')
    )
)

