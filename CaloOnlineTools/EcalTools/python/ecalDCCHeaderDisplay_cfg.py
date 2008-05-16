import FWCore.ParameterSet.Config as cms

process = cms.Process("DCCHEADERDISPLAY")
process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerMapping_cfi")

process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi")

process.load("CaloOnlineTools.EcalTools.ecalDCCHeaderDisplay_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    #untracked uint32 skipEvents = 16000
    fileNames = cms.untracked.vstring('file:/data/franzoni/data/GREN/highRage/fedsOnly30156.root')
)

process.p = cms.Path(process.ecalEBunpacker*process.ecalDccHeaderDisplay)

