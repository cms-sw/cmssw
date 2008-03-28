# The following comments couldn't be translated into the new config version:

#using QCDHLT1jetPE3EventContent

import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from QCDAnalysis.Skimming.QCDHLT1jetPE3EventContent_cff import *
QCDHLT1jetPE3OutputModule = cms.OutputModule("PoolOutputModule",
    QCDHLT1jetPE3EventSelection,
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('QCDHLT1jetPE3'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('QCDHLT1jetPE3.root')
)


