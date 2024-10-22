import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *

# Read constants from DB.
l1csctpdbconfsrc = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    #connect = cms.string('frontier://FrontierDev/CMS_COND_CSC'),
    #connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_21X_CSC'),
    connect =  cms.string('oracle://cms_orcoff_prep/CMS_COND_CSC'),
    #connect = cms.string('sqlite_file:DBL1TPParameters.db'),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('CSCDBL1TPParametersRcd'),
            #tag = cms.string('CSCL1TPParameters')
            tag = cms.string('CSCDBL1TPParameters_hlt')
            #tag = cms.string('CSCL1TPParameters_mc')
        )
    )
)

l1csctpdbconfsrc.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
l1csctpdbconfsrc.DBParameters.authenticationMethod = cms.untracked.uint32(1)

# Reading from DB has precedence over dummy producers (which use constants
# defined in cfi files).
es_prefer_l1csctpdbconfsrc = cms.ESPrefer("PoolDBESSource","l1csctpdbconfsrc")
