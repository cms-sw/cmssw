import FWCore.ParameterSet.Config as cms

process = cms.Process('MyReReco')

process.load("FWCore.MessageService.MessageLogger_cfi")

# load the noise info producer
process.load('RecoMET.METProducers.hcalnoiseinfoproducer_cfi')
process.hcalnoiseinfoproducer.refillRefVectors = cms.bool(True)
process.hcalnoiseinfoproducer.requirePedestals = cms.bool(False)

# run over files
readFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource",fileNames = readFiles)
readFiles.extend( [ 'file:test.root'] )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.MessageLogger.cerr.FwkReport.reportEvery=cms.untracked.int32(10)

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('file:test_refill.root')
                               )

process.p = cms.Path(process.hcalnoiseinfoproducer)
process.e = cms.EndPath(process.out)
