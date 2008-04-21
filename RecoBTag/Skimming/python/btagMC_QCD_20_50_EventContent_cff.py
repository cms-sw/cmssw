import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagMC_QCD_20_50_cfi import *
btagMC_QCD_20-50EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
btagMC_QCD_20-50EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('btagMC_QCD_20-50Path')
    )
)
btagMC_QCD_20-50EventContent.outputCommands.extend(RECOEventContent.outputCommands)

