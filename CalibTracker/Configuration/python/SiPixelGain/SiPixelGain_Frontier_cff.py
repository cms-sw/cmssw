import FWCore.ParameterSet.Config as cms

import copy
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import *
siPixelGainCalibration = copy.deepcopy(poolDBESSource)
siPixelGainCalibration.connect = 'frontier://FrontierProd/CMS_COND_20X_PIXEL'
siPixelGainCalibration.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelGainCalibrationOfflineRcd'),
    tag = cms.string('V2_trivial_TBuffer_mc')
))
siPixelGainCalibration.BlobStreamerName = 'TBufferBlobStreamingService'

