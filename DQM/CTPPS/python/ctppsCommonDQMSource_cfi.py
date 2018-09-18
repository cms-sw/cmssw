import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

ctppsCommonDQMSource = DQMEDAnalyzer('CTPPSCommonDQMSource',
    tagLocalTrackLite = cms.InputTag('ctppsLocalTrackLiteProducer'),
    ctppsmetadata = cms.untracked.InputTag("onlineMetaDataDigis"),
    verbosity = cms.untracked.uint32(0),
)
