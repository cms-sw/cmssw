import FWCore.ParameterSet.Config as cms

from RecoBTag.Skimming.btagCalib_EventContent_cff import *
btagCalibOutputModuleBTAGCALA = cms.OutputModule("PoolOutputModule",
    btagCalibEventSelection,
    BTAGCALAbtagCalibEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagCalibBTAGCALA'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagCalibBTAGCALA.root')
)

btagCalibOutputModuleBTAGCALB = cms.OutputModule("PoolOutputModule",
    btagCalibEventSelection,
    BTAGCALBbtagCalibEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('btagCalibBTAGCALB'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('btagCalibBTAGCALB.root')
)


