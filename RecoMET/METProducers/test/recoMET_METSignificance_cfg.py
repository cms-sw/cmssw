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
        '/store/relval/CMSSW_7_3_0/RelValZMM_13/MINIAODSIM/MCRUN2_73_V7-v1/00000/127CA68E-8981-E411-A524-002590593872.root',
        '/store/relval/CMSSW_7_3_0/RelValZMM_13/MINIAODSIM/MCRUN2_73_V7-v1/00000/56FE228D-8981-E411-9AD8-0025905A6126.root'
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
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

##____________________________________________________________________________||
process.p = cms.Path(
    process.METSignificance
    )

process.e1 = cms.EndPath(
    process.out
    )

##____________________________________________________________________________||
