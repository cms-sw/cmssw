# The following comments couldn't be translated into the new config version:

#FrontierDev/CMS_COND_PIXEL"
import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
#
# Cabling from DB 
#
siPixelCabling = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
siPixelCabling.connect = 'frontier://FrontierDev/CMS_COND_PIXEL'
siPixelCabling.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelFedCablingMapRcd'),
    tag = cms.string('SiPixelFedCablingMap_v9_mc')
))

