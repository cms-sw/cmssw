import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
process = cms.Process("TEST")

##____________________________________________________________________________||
process.load("FWCore.MessageLogger.MessageLogger_cfi")

##____________________________________________________________________________||
process.load("RecoMET.METProducers.HTMET_cfi")

##____________________________________________________________________________||
from RecoMET.METProducers.testInputFiles_cff import recoMETtestInputFiles

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(recoMETtestInputFiles)
    )

##____________________________________________________________________________||
process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('reco_htMET.root'),
    SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_*_*_TEST'
        )
    )

##____________________________________________________________________________||
process.options   = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
process.MessageLogger.cerr.FwkReport.reportEvery = 50
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

##____________________________________________________________________________||
process.p = cms.Path(
    # process.htMetKT4 *
    # process.htMetKT6 *
    # process.htMetIC5 *
    # process.htMetAK7 *
    process.htMetAK5
    )

process.e1 = cms.EndPath(
    process.out
    )

##____________________________________________________________________________||
