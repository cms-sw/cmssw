import FWCore.ParameterSet.Config as cms

process = cms.Process("uncalibRecHitProd")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.EcalTB.readConfiguration2004_v2_cff")

process.load("Configuration.EcalTB.localReco2004_rawData_cff")

process.source = cms.Source("Ecal2004TBSource",
    maxEvents = cms.untracked.int32(1000),
    fileNames = cms.untracked.vstring('file:/u1/meridian/data/h4/2004/ecs73276')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('tbuncalibhits.root')
)

process.p = cms.Path(process.getCond*process.localReco2004_rawData)
process.ep = cms.EndPath(process.out)

