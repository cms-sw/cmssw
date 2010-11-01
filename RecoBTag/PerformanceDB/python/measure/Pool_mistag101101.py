import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourceMistag101101 = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGSSVHEMtable_v3_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHEMtable_v3_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGSSVHEMwp_v3_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHEMwp_v3_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGSSVHPTtable_v3_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHPTtable_v3_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGSSVHPTwp_v3_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHPTwp_v3_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHELtable_v3_offline'),
    label = cms.untracked.string('BTagMISTAGTCHELtable_v3_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHELwp_v3_offline'),
    label = cms.untracked.string('BTagMISTAGTCHELwp_v3_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHEMtable_v3_offline'),
    label = cms.untracked.string('BTagMISTAGTCHEMtable_v3_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHEMwp_v3_offline'),
    label = cms.untracked.string('BTagMISTAGTCHEMwp_v3_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHPTtable_v3_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPTtable_v3_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHPTwp_v3_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPTwp_v3_offline')
    ),
))
PoolDBESSourceMistag101101.connect = 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS'
                              
                              
                              
