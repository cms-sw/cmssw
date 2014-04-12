import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from QCDAnalysis.Skimming.QCDHLT1jetPE7EventContent_cff import *
QCDHLT1jetPE7OutputModule = cms.OutputModule("PoolOutputModule",
    #using QCDHLT1jetPE7EventContent
    QCDHLT1jetPE7EventSelection,
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('QCDHLT1jetPE7'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('QCDHLT1jetPE7.root')
)


