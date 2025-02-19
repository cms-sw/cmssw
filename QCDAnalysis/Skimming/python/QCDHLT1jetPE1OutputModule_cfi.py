import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from QCDAnalysis.Skimming.QCDHLT1jetPE1EventContent_cff import *
QCDHLT1jetPE1OutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    #using QCDHLT1jetPE1EventContent
    QCDHLT1jetPE1EventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('QCDHLT1jetPE1'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('QCDHLT1jetPE1.root')
)


