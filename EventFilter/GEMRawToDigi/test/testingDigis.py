import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1)) # 370000

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#      'file:/afs/cern.ch/user/c/cepeda/public/CosmicStand/DigisRun000003.root'
      'file:DigisRun000006.root'
#        'file:/afs/cern.ch/user/c/cepeda/public/forJason_CSRUN0003.root'
#      'file:myGEMFedRawDataFile.root'
#      'file:test2.root'
    )
)

process.check = cms.EDAnalyzer("GEMDigiReader",
      InputLabel = cms.InputTag("gemDigis"))

process.recHit = cms.EDProducer("GEMRecHitProducer",
      gemDigiLabel = cms.InputTag("gemDigis"))

  
process.p = cms.Path(process.check)

