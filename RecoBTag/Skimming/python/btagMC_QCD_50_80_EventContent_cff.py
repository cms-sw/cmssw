import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagMC_QCD_50_80_cfi import *
btagMC_QCD_50-80EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
btagMC_QCD_50-80EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('btagMC_QCD_50-80Path')
    )
)
btagMC_QCD_50-80EventContent.outputCommands.extend(RECOEventContent.outputCommands)

