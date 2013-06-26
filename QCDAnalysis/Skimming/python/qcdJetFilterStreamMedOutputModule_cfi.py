import FWCore.ParameterSet.Config as cms

from QCDAnalysis.Skimming.qcdJetFilterStreamMed_EventContent_cff import *
from Configuration.EventContent.EventContent_cff import *

qcdJetFilterStreamMedOutputModule = cms.OutputModule("PoolOutputModule",
    qcdJetFilterStreamMedEventSelection,
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('qcdJetFilterStreamMedPath'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('qcdJetFilterStreamMed.root')
)


