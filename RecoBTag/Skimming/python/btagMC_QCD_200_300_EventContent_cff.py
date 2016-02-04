import FWCore.ParameterSet.Config as cms



from Configuration.EventContent.EventContent_cff import *

from RecoBTag.Skimming.btagMC_QCD_200_300_cfi import *
btagMC_QCD_200_300EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
btagMC_QCD_200_300EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('btagMC_QCD_200_300Path')
    )
)

