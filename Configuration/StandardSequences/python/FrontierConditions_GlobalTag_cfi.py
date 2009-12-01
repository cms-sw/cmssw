import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *

GlobalTag = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'),
#    connect = cms.string('sqlite_fip:CondCore/TagCollection/data/GlobalTag.db'), #For use during release integration
    globaltag = cms.string('UNSPECIFIED::All'),
    RefreshEachRun=cms.untracked.bool(False),
    DumpStat=cms.untracked.bool(False),
    pfnPrefix=cms.untracked.string(''),   
    pfnPostfix=cms.untracked.string('')
)
