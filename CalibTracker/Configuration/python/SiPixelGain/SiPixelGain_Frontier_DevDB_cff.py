# The following comments couldn't be translated into the new config version:

#FrontierDev/CMS_COND_PIXEL"
import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
siPixelGainCalibration = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
siPixelGainCalibration.connect = 'frontier://FrontierDev/CMS_COND_PIXEL'
siPixelGainCalibration.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelGainCalibrationOfflineRcd'),
    tag = cms.string('V2_trivial_TBuffer_mc')
))
siPixelGainCalibration.BlobStreamerName = 'TBufferBlobStreamingService'

