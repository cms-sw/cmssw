import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *

process = cms.Process("SkimTest")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 10

## Input files
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_2/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2-v1/0007/E834B497-B178-DE11-9663-001D09F2447F.root'
    )
                            )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(500))

# Define the Trigger Bits to skim on
process.load("DiffractiveForwardAnalysis.Skimming.ExampleHLTFilter_cfi")

process.output = cms.OutputModule("PoolOutputModule",
#                                  AODSIMEventContent,
                                  outputCommands = cms.untracked.vstring("keep *"),
                                  fileName = cms.untracked.string('file:/tmp/jjhollar/tau.aodskim.root'),
                                  dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('SKIM'),
    filterName = cms.untracked.string("hltFilter")
    ),
                                  )

#process.output.outputCommands.extend(AODEventContent.outputCommands)

process.p = cms.Path(
    process.hltFilter +
    process.output
    )
