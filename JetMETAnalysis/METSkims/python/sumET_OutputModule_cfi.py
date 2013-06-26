import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.METSkims.sumET_EventContent_cff import *
from JetMETAnalysis.METSkims.AODSIMSumET_EventContent_cff import *
sumETOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    sumETEventSelection,
    AODSIMSumETEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('sumET_AODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('sumET_AODSIM.root')
)


