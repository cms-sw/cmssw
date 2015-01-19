import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
process = cms.Process("TEST")

##____________________________________________________________________________||
process.load("FWCore.MessageLogger.MessageLogger_cfi")

##____________________________________________________________________________||
process.load("RecoMET/METProducers.METSignificance_cfi")
process.load("RecoMET/METProducers.METSignificanceParams_cfi")

##____________________________________________________________________________||
from RecoMET.METProducers.testInputFiles_cff import recoMETtestInputFiles
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
       ### MINIAODSIM
       '/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/MINIAODSIM/PU25ns_PRE_LS172_V15-v1/00000//A652D13A-335F-E411-90BA-02163E008D01.root'
       )
    )

##____________________________________________________________________________||
process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('recoMET_METSignificance.root'),
    SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_*_*_TEST'
        )
    )

##____________________________________________________________________________||
process.options   = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
process.MessageLogger.cerr.FwkReport.reportEvery = 50
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

##____________________________________________________________________________||
process.p = cms.Path(
    process.METSignificance
    )

process.e1 = cms.EndPath(
    process.out
    )

##____________________________________________________________________________||
