import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *
#from L1TriggerConfig.L1CSCTPConfigProducers.L1CSCTriggerPrimitivesConfig_cfi import *

# Read constants from DB.
l1csctpdbconfsrc = cms.ESSource("PoolDBESSource",
    CondDBCommon,
    DBParameters = cms.PSet(
      #authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
      authenticationMethod = cms.untracked.uint32(1)
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
      record = cms.string('CSCL1TPParametersRcd'),
      tag = cms.string('CSCL1TPParameters')
      #type = cms.string('L1CSCTPParameters')
    )),
    connect = cms.string('frontier://FrontierDev/CMS_COND_CSC')
    #connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_21X_CSC')
)

# Reading from DB has precedence over dummy producers (which use constants
# defined in cfi files).
es_prefer_l1csctpdbconfsrc = cms.ESPrefer("PoolDBESSource","l1csctpdbconfsrc")
