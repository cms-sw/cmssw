import FWCore.ParameterSet.Config as cms

SiStripMonitorHLT = cms.EDAnalyzer("SiStripMonitorHLT",
    HLTProducer = cms.string('trigger')
)
