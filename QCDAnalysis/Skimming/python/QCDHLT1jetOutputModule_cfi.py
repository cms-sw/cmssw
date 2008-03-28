# The following comments couldn't be translated into the new config version:

#using QCDHLT1jetEventContent

import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from QCDAnalysis.Skimming.QCDHLT1jetEventContent_cff import *
QCDHLT1jetOutputModule = cms.OutputModule("PoolOutputModule",
    QCDHLT1jetEventSelection,
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('QCDHLT1jet'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('QCDHLT1jet.root')
)


