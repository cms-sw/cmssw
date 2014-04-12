import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *

from RecoBTag.Skimming.btagMC_QCD_50_80_cfi import *
btagMC_QCD_50_80EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
btagMC_QCD_50_80EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('btagMC_QCD_50_80Path')
    )
)
btagMC_QCD_50_80EventContent.outputCommands.extend(RECOEventContent.outputCommands)

