import FWCore.ParameterSet.Config as cms

SiStripMonitorFilter = cms.EDAnalyzer("SiStripMonitorFilter",
    FilterProducer = cms.string('ClusterMTCCFilter')
)
