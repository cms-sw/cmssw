import FWCore.ParameterSet.Config as cms

import copy
from CalibTracker.Configuration.Common.PoolDBESSource_cfi import *
siPixelCabling = copy.deepcopy(poolDBESSource)
siPixelCabling.connect = 'sqlite_fip:CondCore/SQLiteData/data/siPixelCabling200.db'
siPixelCabling.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelFedCablingMapRcd'),
    tag = cms.string('SiPixelFedCablingMap_v9')
))

