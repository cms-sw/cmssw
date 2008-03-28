import FWCore.ParameterSet.Config as cms

# To be used on SQLite files.  AFS access not required.
from CondCore.DBCommon.CondDBSetup_cfi import *
DTCabling = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTReadOutMappingRcd'),
        tag = cms.string('map_fix17X')
    )),
    connect = cms.string('sqlite_fip:CondCore/SQLiteData/data/DTFullMap_fix17X.db')
)


