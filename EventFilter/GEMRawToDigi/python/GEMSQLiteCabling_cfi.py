import FWCore.ParameterSet.Config as cms

from CondCore.CondDB.CondDB_cfi import *
CondDB.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
CondDB.connect = cms.string('sqlite_fip:EventFilter/GEMRawToDigi/data/GEMEMap.db')
#from CondCore.DBCommon.CondDBSetup_cfi import *
#CondDBSetup.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
GEMCabling = cms.ESSource("PoolDBESSource",
    CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('GEMEMapRcd'),
        tag = cms.string('GEMEMap_v2')
    )),
)


