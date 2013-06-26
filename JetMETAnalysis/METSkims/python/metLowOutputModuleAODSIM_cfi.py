import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.METSkims.metLow_EventContent_cff import *
from JetMETAnalysis.METSkims.AODSIMMetLow_EventContent_cff import *
metLowOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    metLowEventSelection,
    AODSIMMetLowEventContent,
    fileName = cms.untracked.string('metLow_AODSIM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('metLow_AODSIM'),
        dataTier = cms.untracked.string('USER')
    )
)


