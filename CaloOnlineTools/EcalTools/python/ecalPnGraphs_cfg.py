import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTPNGRAPHDUMPER")
process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerMapping_cfi")

process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi")

process.load("CaloOnlineTools.EcalTools.ecalPnGraphs_cfi")

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    skipEvents = cms.untracked.uint32(0),
    #untracked vstring fileNames = { 'file:/data/kkaadze/work/DQM/dataFiles/P5_Co.00028065.A.0.0.root' }
    fileNames = cms.untracked.vstring('file:........')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.p = cms.Path(process.ecalEBunpacker*process.ecalPnGraphs)

