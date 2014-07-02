import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
process = cms.Process("TEST")

##____________________________________________________________________________||
process.load("FWCore.MessageLogger.MessageLogger_cfi")

##____________________________________________________________________________||
process.load("RecoMET/METProducers.PFMET_cfi")
process.load("RecoMET/METProducers.METSigParams_cfi")

##____________________________________________________________________________||
from RecoMET.METProducers.testInputFiles_cff import recoMETtestInputFiles
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(recoMETtestInputFiles)
    )

##____________________________________________________________________________||
process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('recoMET_pfMet.root'),
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
process.pfMetWithSignificance = process.pfMet.clone(
    process.METSignificance_params,
    calculateSignificance = cms.bool(True),
    jets = cms.InputTag("ak5PFJets")
    )

##____________________________________________________________________________||
process.p = cms.Path(
    process.pfMetWithSignificance *
    process.pfMet
    )

process.e1 = cms.EndPath(
    process.out
    )

##____________________________________________________________________________||
