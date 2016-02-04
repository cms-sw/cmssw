import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
CondDBSetup.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
RPCCabling = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('RPCEMapRcd'),
        tag = cms.string('RPCEMap_v2')
    )),
#    connect = cms.string('sqlite_file:RPCEMap_181007.db')
    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_36X_RPC')
)


