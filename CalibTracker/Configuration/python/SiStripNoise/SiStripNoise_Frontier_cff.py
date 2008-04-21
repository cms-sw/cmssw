import FWCore.ParameterSet.Config as cms

import copy
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import *
siStripNoise = copy.deepcopy(poolDBESSource)
siStripNoise.connect = 'frontier://FrontierProd/CMS_COND_20X_STRIP'
siStripNoise.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiStripNoisesRcd'),
    tag = cms.string('SiStripNoise_Fake_PeakMode_20X')
))
siStripNoise.BlobStreamerName = 'TBufferBlobStreamingService'

