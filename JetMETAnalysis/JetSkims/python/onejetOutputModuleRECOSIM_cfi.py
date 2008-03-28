import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.onejet_EventContent_cff import *
from JetMETAnalysis.JetSkims.RECOSIMOneJetEventContent_cff import *
onejetOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    onejetEventSelection,
    RECOSIMOneJetEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('onejet_RECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('onejet_RECOSIM.root')
)


