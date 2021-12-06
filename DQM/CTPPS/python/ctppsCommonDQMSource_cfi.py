import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

ctppsCommonDQMSource = DQMEDAnalyzer('CTPPSCommonDQMSource',
    ctppsmetadata = cms.untracked.InputTag("onlineMetaDataDigis"),
    tagLocalTrackLite = cms.InputTag('ctppsLocalTrackLiteProducer'),
    tagRecoProtons = cms.InputTag("ctppsProtons", "multiRP"),

    makeProtonRecoPlots = cms.bool(True),

    perLSsaving = cms.untracked.bool(False), #driven by DQMServices/Core/python/DQMStore_cfi.py

    verbosity = cms.untracked.uint32(0),
)
