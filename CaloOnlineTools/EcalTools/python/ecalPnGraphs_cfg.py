import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTPNGRAPHDUMPER")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("CaloOnlineTools.EcalTools.ecalPnGraphs_cfi")

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    skipEvents = cms.untracked.uint32(0),
    #untracked vstring fileNames = { 'file:/data/kkaadze/work/DQM/dataFiles/P5_Co.00028065.A.0.0.root' }
    #fileNames = cms.untracked.vstring('file:........')
    fileNames = cms.untracked.vstring('file:/data/scooper/data/postBeam/laser/ecal_local.00063460.0001.A.storageManager.0.0000.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.p = cms.Path(process.ecalEBunpacker*process.ecalPnGraphs)

