import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from QCDAnalysis.Skimming.qcdJetFilterStreamHi_EventContent_cff import *
qcdJetFilterStreamHiOutputModule = cms.OutputModule("PoolOutputModule",
    qcdJetFilterStreamHiEventSelection,
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('qcdJetFilterStreamHiPath'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('qcdJetFilterStreamHi.root')
)


