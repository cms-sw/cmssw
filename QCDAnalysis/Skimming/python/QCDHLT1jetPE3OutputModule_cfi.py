import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from QCDAnalysis.Skimming.QCDHLT1jetPE3EventContent_cff import *
QCDHLT1jetPE3OutputModule = cms.OutputModule("PoolOutputModule",
    #using QCDHLT1jetPE3EventContent
    QCDHLT1jetPE3EventSelection,
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('QCDHLT1jetPE3'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('QCDHLT1jetPE3.root')
)


