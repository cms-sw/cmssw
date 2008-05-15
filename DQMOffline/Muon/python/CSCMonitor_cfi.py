import FWCore.ParameterSet.Config as cms

cscMonitor = cms.EDFilter("CSCOfflineMonitor",
    outputFileName = cms.string('test.root'),
    saveHistos = cms.bool(False)
)


