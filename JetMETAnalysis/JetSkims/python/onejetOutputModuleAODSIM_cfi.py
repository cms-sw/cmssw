import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.onejet_EventContent_cff import *
from JetMETAnalysis.JetSkims.AODSIMOneJetEventContent_cff import *
onejetOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    AODSIMOneJetEventContent,
    onejetEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('onejet_AODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('onejet_AODSIM.root')
)


