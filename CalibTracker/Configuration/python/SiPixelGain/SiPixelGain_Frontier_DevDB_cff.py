import FWCore.ParameterSet.Config as cms

import copy
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import *
siPixelGainCalibration = copy.deepcopy(poolDBESSource)
siPixelGainCalibration.connect = 'frontier://FrontierDev/CMS_COND_PIXEL'
siPixelGainCalibration.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelGainCalibrationOfflineRcd'),
    tag = cms.string('V2_trivial_TBuffer_mc')
))
siPixelGainCalibration.BlobStreamerName = 'TBufferBlobStreamingService'

