import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
process = cms.Process("TEST")

##____________________________________________________________________________||
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("RecoJets.JetProducers.PFClustersForJets_cff")
process.load("RecoMET/METProducers/PFClusterMET_cfi")

##____________________________________________________________________________||
from RecoMET.METProducers.testInputFiles_cff import recoMETtestInputFiles

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(recoMETtestInputFiles)
    )

##____________________________________________________________________________||
process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('reco_pfClusterMET.root'),
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
    process.particleFlowClusterHF*
    process.particleFlowClusterHO*
    process.pfClusterRefsForJetsHCAL*
    process.pfClusterRefsForJetsECAL*
    process.pfClusterRefsForJetsHF*
    process.pfClusterRefsForJetsHO*
    process.pfClusterRefsForJets *
    process.pfClusterMet
    )

process.e1 = cms.EndPath(
    process.out
    )

##____________________________________________________________________________||
