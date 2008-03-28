import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
RPCCabling = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('RPCEMapRcd'),
        tag = cms.string('RPCEMap_v2')
    )),
    connect = cms.string('sqlite_fip:CondCore/SQLiteData/data/RPCEMap_181007.db')
)


