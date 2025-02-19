import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTDUMPER")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("CaloOnlineTools.EcalTools.ecalDigiDisplay_cfi")

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring('file:.........')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.counter = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.ecalEBunpacker*process.ecalDigiDisplay)
process.end = cms.EndPath(process.counter)

