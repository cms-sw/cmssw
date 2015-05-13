import FWCore.ParameterSet.Config as cms

process = cms.Process("skim")

process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond_condDBv2 import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step3QCD.root')
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('Configuration/EventContent/EventContent_cff')

#hotline filters
process.load("Calibration.Hotline.hotlineSkims_cff")
process.load("Calibration.Hotline.hotlineSkims_Output_cff")

process.out = cms.OutputModule("PoolOutputModule",
    SelectEvents = process.OutALCARECOHotline.SelectEvents,
    outputCommands = process.OutALCARECOHotline.outputCommands,
    fileName = cms.untracked.string('prova.root')
)

process.e = cms.EndPath(process.out)
