import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripZeroSuppression.DefaultAlgorithms_cff import *

SiStripFEDEmulator = cms.EDProducer("SiStripFEDEmulatorModule",
    SpyReorderedDigisTag = cms.InputTag('SiStripSpyDigiConverter','Reordered'),
    SpyVirginRawDigisTag = cms.InputTag('SiStripSpySigiConverter','VirginRaw'),
    ByModule = cms.bool(True),
    Algorithms = DefaultAlgorithms
)
