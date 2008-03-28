import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagMC_ttbar_cfi import *
btagMC_ttbarEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
btagMC_ttbarEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('btagMC_ttbarPath')
    )
)
btagMC_ttbarEventContent.outputCommands.extend(RECOEventContent.outputCommands)

