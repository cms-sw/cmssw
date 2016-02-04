import FWCore.ParameterSet.Config as cms

process = cms.Process("uncalibRecHitsProd")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("EventFilter.EcalTBRawToDigi.ecalTBunpack_cfi")

process.load("Configuration.EcalTB.readConfiguration2006_v1_fromFrontier_cff")

process.load("Configuration.EcalTB.localReco2006_rawData_cff")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.Timing = cms.Service("Timing")

process.source = cms.Source("PoolSource",
    maxEvents = cms.untracked.int32(500),
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/archive/ecal/h4tb.pool/h4b.00017056.A.0.0.root'),
    isBinary = cms.untracked.bool(True)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('hits.root')
)

process.p = cms.Path(process.getCond*process.ecalTBunpack*process.localReco2006_rawData)
process.ep = cms.EndPath(process.out)

