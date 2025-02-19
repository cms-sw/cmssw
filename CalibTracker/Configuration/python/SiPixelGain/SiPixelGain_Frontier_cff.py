# The following comments couldn't be translated into the new config version:

#FrontierProd/CMS_COND_20X_PIXEL"
import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
siPixelGainCalibration = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
siPixelGainCalibration.connect = 'frontier://FrontierProd/CMS_COND_20X_PIXEL'
siPixelGainCalibration.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelGainCalibrationOfflineRcd'),
    tag = cms.string('V2_trivial_TBuffer_mc')
))
siPixelGainCalibration.BlobStreamerName = 'TBufferBlobStreamingService'

