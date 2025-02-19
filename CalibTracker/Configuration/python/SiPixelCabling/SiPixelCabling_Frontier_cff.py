# The following comments couldn't be translated into the new config version:

#FrontierProd/CMS_COND_20X_PIXEL"
import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
#
# Cabling from DB 
#
siPixelCabling = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
siPixelCabling.connect = 'frontier://FrontierProd/CMS_COND_20X_PIXEL'
siPixelCabling.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelFedCablingMapRcd'),
    tag = cms.string('SiPixelFedCablingMap_v9_mc')
))

