import FWCore.ParameterSet.Config as cms

SiStripMonitorHLT = DQMStep1Module('SiStripMonitorHLT',
    HLTProducer = cms.string('trigger')
)
