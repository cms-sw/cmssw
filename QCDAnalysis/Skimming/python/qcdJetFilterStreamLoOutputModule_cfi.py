import FWCore.ParameterSet.Config as cms

from QCDAnalysis.Skimming.qcdJetFilterStreamLo_EventContent_cff import *
from Configuration.EventContent.EventContent_cff import *

qcdJetFilterStreamLoOutputModule = cms.OutputModule("PoolOutputModule",
    qcdJetFilterStreamLoEventSelection,
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('qcdJetFilterStreamLoPath'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('qcdJetFilterStreamLo.root')
)


