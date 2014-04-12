import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from QCDAnalysis.Skimming.QCDHLT1jetPE5EventContent_cff import *
QCDHLT1jetPE5OutputModule = cms.OutputModule("PoolOutputModule",
    #using QCDHLT1jetPE5EventContent
    QCDHLT1jetPE5EventSelection,
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('QCDHLT1jetPE5'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('QCDHLT1jetPE5.root')
)


