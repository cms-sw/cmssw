import FWCore.ParameterSet.Config as cms

cscMonitor = cms.EDFilter("CSCOfflineMonitor",
    stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    outputFileName = cms.string('test.root'),
    cscRecHitTag = cms.InputTag("csc2DRecHits"),
    saveHistos = cms.bool(False),
    wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    cscSegTag = cms.InputTag("cscSegments")
)



