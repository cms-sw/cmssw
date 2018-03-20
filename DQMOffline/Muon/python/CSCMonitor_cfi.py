import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
cscMonitor = DQMEDAnalyzer('CSCOfflineMonitor',
    stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    outputFileName = cms.string('test.root'),
    cscRecHitTag = cms.InputTag("csc2DRecHits"),
    saveHistos = cms.bool(False),
    wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    alctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCALCTDigi"),
    clctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCLCTDigi"),
    cscSegTag = cms.InputTag("cscSegments"),
    FEDRawDataCollectionTag = cms.InputTag("rawDataCollector")
)



