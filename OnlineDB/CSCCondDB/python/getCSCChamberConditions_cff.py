import FWCore.ParameterSet.Config as cms

#used for chamber DB conditions
from CondCore.DBCommon.CondDBSetup_cfi import *
cscConditions = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    #    string connect = "frontier://FrontierDev/CMS_COND_ALIGNMENT"
    siteLocalConfig = cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCChamberMapRcd'),
        tag = cms.string('CSCChamberMap')
    ), 
        cms.PSet(
            record = cms.string('CSCChamberIndexRcd'),
            tag = cms.string('CSCChamberIndex')
        ), 
        cms.PSet(
            record = cms.string('CSCCrateMapRcd'),
            tag = cms.string('CSCCrateMap')
        ), 
        cms.PSet(
            record = cms.string('CSCDDUMapRcd'),
            tag = cms.string('CSCDDUMap')
        )),
    messagelevel = cms.untracked.uint32(0),
    timetype = cms.string('runnumber'),
    connect = cms.string('oracle://cms_orcoff_int2r/CMS_COND_CSC'), ##cms_orcoff_int2r/CMS_COND_CSC"

    authenticationMethod = cms.untracked.uint32(1)
)

cscConditions.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

