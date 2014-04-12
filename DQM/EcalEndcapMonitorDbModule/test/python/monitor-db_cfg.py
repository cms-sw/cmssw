import FWCore.ParameterSet.Config as cms

process = cms.Process("MONITOR")

process.load("DQMServices.Core.DQM_cfg")

process.ecalDbMonitor = cms.EDAnalyzer("EcalEndcapMonitorDbModule",
    htmlDir = cms.untracked.string('.'),
    prefixME = cms.untracked.string('EcalEndcap')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("EmptySource")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noTimeStamps = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('cout')
)

process.p = cms.Path(process.ecalDbMonitor)

process.DQM.collectorHost = 'localhost'

