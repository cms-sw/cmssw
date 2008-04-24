# The following comments couldn't be translated into the new config version:

#FrontierDev/CMS_COND_STRIP"
import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
siStripNoise = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
siStripNoise.connect = 'frontier://FrontierDev/CMS_COND_STRIP'
siStripNoise.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiStripNoisesRcd'),
    tag = cms.string('SiStripNoise_Fake_PeakMode_20X')
))
siStripNoise.BlobStreamerName = 'TBufferBlobStreamingService'

