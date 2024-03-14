# The following comments couldn't be translated into the new config version:

#FrontierDev/CMS_COND_TIF_PIXELS"
import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
#
# Cabling from DB 
#
siPixelCabling = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
siPixelCabling.connect = 'frontier://FrontierDev/CMS_COND_TIF_PIXELS'
siPixelCabling.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelFedCablingMapRcd'),
    tag = cms.string('SiPixelFedCablingMap_v9')
))

# dummy dummy
