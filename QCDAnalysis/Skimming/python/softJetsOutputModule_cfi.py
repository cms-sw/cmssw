import FWCore.ParameterSet.Config as cms

from QCDAnalysis.Skimming.softJetsEventContent_cfi import *
softJetsOutputModule = cms.OutputModule("PoolOutputModule",
    softJetsEventSelection,
    softJetsEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('softJets'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('softJets.root')
)


