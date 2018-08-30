import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *

siStripMergeZeroSuppression = cms.EDProducer("SiStripMergeZeroSuppression",
    Algorithms     = DefaultAlgorithms,
    DigisToMergeZS = cms.InputTag('siStripDigis','ZeroSuppressed'),
    DigisToMergeVR = cms.InputTag('siStripVRDigis','VirginRaw'),
)
