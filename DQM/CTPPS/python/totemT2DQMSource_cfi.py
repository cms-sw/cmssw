import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
totemT2DQMSource = DQMEDAnalyzer('TotemT2DQMSource',
    digisTag = cms.InputTag('totemT2Digis', 'TotemT2'),
    rechitsTag = cms.InputTag('totemT2RecHits'),
    nbinsx = cms.uint32(25),
    nbinsy = cms.uint32(25),
    windowsNum = cms.uint32(8),

    perLSsaving = cms.untracked.bool(False), #driven by DQMServices/Core/python/DQMStore_cfi.py
)
