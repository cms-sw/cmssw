import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing ("analysis")
options.parseArguments()

process = cms.Process("ZMuMuMassConstraintParameterFinder")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles)
    )

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string(options.outputFile),
    closeFileFast = cms.untracked.bool(True)
)

process.load("Alignment.MillePedeAlignmentAlgorithm.zMuMuMassConstraintParameterFinder_cfi")

process.p = cms.Path(process.zMuMuMassConstraintParameterFinder)
