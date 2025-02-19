import FWCore.ParameterSet.Config as cms

import CalibTracker.Configuration.Common.PoolDBESSource_cfi
# Using SQLite cabling maps.  No AFS required.
siPixelCabling = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()
siPixelCabling.connect = 'sqlite_fip:CondCore/SQLiteData/data/siPixelCabling200.db'
siPixelCabling.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelFedCablingMapRcd'),
    tag = cms.string('SiPixelFedCablingMap_v9')
))

