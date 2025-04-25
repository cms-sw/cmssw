import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
# To be used on SQLite files.  AFS access not required.
DTCabling = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTReadOutMappingRcd'),
        tag = cms.string('map_fix17X')
    )),
    connect = cms.string('sqlite_fip:CondCore/SQLiteData/data/DTFullMap_fix17X.db')
)


