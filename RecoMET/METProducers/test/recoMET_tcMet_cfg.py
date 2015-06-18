import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
process = cms.Process("TEST")

##____________________________________________________________________________||
process.load("FWCore.MessageLogger.MessageLogger_cfi")

##____________________________________________________________________________||
process.load("RecoMET.METProducers.MuonTCMETValueMapProducer_cff")
process.load("RecoMET.METProducers.TCMET_cfi")
process.load("RecoMET.METProducers.CaloMET_cfi")
process.load("RecoMET.Configuration.RecoTCMET_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff')
process.load("RecoLocalCalo.Configuration.RecoLocalCalo_cff")

##____________________________________________________________________________||
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

##____________________________________________________________________________||
from RecoMET.METProducers.testInputFiles_cff import recoMETtestInputFiles

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(recoMETtestInputFiles)
    )

##____________________________________________________________________________||
process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('recoMET_tcMet.root'),
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
process.tcMetCST = process.tcMet.clone()
process.tcMetCST.correctShowerTracks = cms.bool(True)

process.tcMetRft2 = process.tcMet.clone()
process.tcMetRft2.rf_type = cms.int32(2)

process.tcMetVedu = process.tcMet.clone()
process.tcMetVedu.vetoDuplicates = cms.bool(True)

##____________________________________________________________________________||
process.p = cms.Path(
    process.particleFlowCluster *
    process.muonTCMETValueMapProducer *
    process.tcMet *
    process.tcMetCST *
    process.tcMetRft2 *
    process.tcMetVedu *
    process.tcMetWithPFclusters
    )

process.e1 = cms.EndPath(
    process.out
    )

##____________________________________________________________________________||
