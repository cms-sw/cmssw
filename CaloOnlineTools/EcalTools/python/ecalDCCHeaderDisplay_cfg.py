import FWCore.ParameterSet.Config as cms

process = cms.Process("DCCHEADERDISPLAY")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("CaloOnlineTools.EcalTools.ecalDCCHeaderDisplay_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    #untracked uint32 skipEvents = 16000
    fileNames = cms.untracked.vstring('file:/data/scooper/data/cruzet3/7E738216-584D-DD11-9209-000423D6AF24.root')
)

process.p = cms.Path(process.ecalEBunpacker*process.ecalDccHeaderDisplay)

