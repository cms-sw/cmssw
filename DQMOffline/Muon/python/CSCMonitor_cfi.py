import FWCore.ParameterSet.Config as cms

cscMonitor = cms.EDFilter("CSCOfflineMonitor",
    outputFileName = cms.string('test.root'),
    saveHistos = cms.bool(False),
    wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    cscRecHitTag = cms.InputTag("csc2DRecHits"),
    cscSegTag    = cms.InputTag("cscSegments")
)



