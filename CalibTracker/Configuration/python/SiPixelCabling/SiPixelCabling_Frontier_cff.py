import FWCore.ParameterSet.Config as cms

import copy
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import *
siPixelCabling = copy.deepcopy(poolDBESSource)
siPixelCabling.connect = 'frontier://FrontierProd/CMS_COND_20X_PIXEL'
siPixelCabling.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelFedCablingMapRcd'),
    tag = cms.string('SiPixelFedCablingMap_v9_mc')
))

