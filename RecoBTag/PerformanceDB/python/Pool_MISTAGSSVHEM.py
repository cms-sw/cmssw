import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSource = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGSSVHEMtable_v2_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHEMtable_v2_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGSSVHEMwp_v2_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHEMwp_v2_offline')
    ),
))
PoolDBESSource.connect = 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS'
