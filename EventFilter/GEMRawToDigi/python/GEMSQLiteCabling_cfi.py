import FWCore.ParameterSet.Config as cms

from CondCore.CondDB.CondDB_cfi import *
CondDB.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
CondDB.connect = cms.string('sqlite_fip:EventFilter/GEMRawToDigi/data/GEMELMap.db')
#from CondCore.DBCommon.CondDBSetup_cfi import *
#CondDBSetup.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
GEMCabling = cms.ESSource("PoolDBESSource",
    CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('GEMELMapRcd'),
        tag = cms.string('GEMELMap_v2')
    )),
)


