import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
SiStripDBCabling = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripFedCablingRcd'),
        tag = cms.string('SiStripFedCabling_v1')
    )),
    connect = cms.string('sqlite_fip:CondCore/SQLiteData/data/sistripfedcablingCSA07.db')
)


