import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *

from RecoBTag.Skimming.btagMC_QCD_20_50_cfi import *
btagMC_QCD_20_50EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
btagMC_QCD_20_50EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('btagMC_QCD_20_50Path')
    )
)
btagMC_QCD_20_50EventContent.outputCommands.extend(RECOEventContent.outputCommands)

