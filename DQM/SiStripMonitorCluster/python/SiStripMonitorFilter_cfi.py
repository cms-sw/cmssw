import FWCore.ParameterSet.Config as cms

SiStripMonitorFilter = DQMStep1Module('SiStripMonitorFilter',
    FilterProducer = cms.string('ClusterMTCCFilter')
)
