import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTGRAPHDUMPER")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

process.load("CaloOnlineTools.EcalTools.ecalPedHists_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    #untracked uint32 skipEvents = 16000
    #untracked vstring fileNames = {'file:/data/scooper/data/P5_Co-07/P5_Co.00027909.A.0.0.root'}
    #fileNames = cms.untracked.vstring('file:/data/scooper/data/grea07/40792E58-B757-DC11-8AB2-001617E30F46.root')
    fileNames = cms.untracked.vstring('file:/data/scooper/data/cruzet3/7E738216-584D-DD11-9209-000423D6AF24.root')
)

process.p = cms.Path(process.ecalEBunpacker*process.ecalPedHists)

