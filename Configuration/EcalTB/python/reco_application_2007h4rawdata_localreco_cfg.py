import FWCore.ParameterSet.Config as cms

process = cms.Process("uncalibRecHitsProd")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.EcalTB.unpacker2007h4_cff")

process.load("Configuration.EcalTB.readConfigurationH4_2007_v0_cff")

process.load("Configuration.EcalTB.localReco2007h4_rawData_cff")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.Timing = cms.Service("Timing")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/cmsrm/pc18/meridian/h4_2007/h4b.00019738.A.0.0.root')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('hits.root'),
    outputCommands = cms.untracked.vstring("keep *",
                                           "drop FEDRawData*_*_*_*"
                                           )
)

process.p = cms.Path(process.ecalTBunpack*process.localReco2007h4_rawData)
process.ep = cms.EndPath(process.out)
