import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSourceMistag110118 = cms.ESSource("PoolDBESSource",
                              CondDBCommon,
                              toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJBPLtable_v5_offline'),
    label = cms.untracked.string('BTagMISTAGJBPLtable_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJBPLwp_v5_offline'),
    label = cms.untracked.string('BTagMISTAGJBPLwp_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJBPMtable_v5_offline'),
    label = cms.untracked.string('BTagMISTAGJBPMtable_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJBPMwp_v5_offline'),
    label = cms.untracked.string('BTagMISTAGJBPMwp_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJBPTtable_v5_offline'),
    label = cms.untracked.string('BTagMISTAGJBPTtable_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJBPTwp_v5_offline'),
    label = cms.untracked.string('BTagMISTAGJBPTwp_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJPLtable_v5_offline'),
    label = cms.untracked.string('BTagMISTAGJPLtable_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJPLwp_v5_offline'),
    label = cms.untracked.string('BTagMISTAGJPLwp_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJPMtable_v5_offline'),
    label = cms.untracked.string('BTagMISTAGJPMtable_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJPMwp_v5_offline'),
    label = cms.untracked.string('BTagMISTAGJPMwp_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGJPTtable_v5_offline'),
    label = cms.untracked.string('BTagMISTAGJPTtable_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGJPTwp_v5_offline'),
    label = cms.untracked.string('BTagMISTAGJPTwp_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGSSVHEMtable_v5_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHEMtable_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGSSVHEMwp_v5_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHEMwp_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGSSVHPTtable_v5_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHPTtable_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGSSVHPTwp_v5_offline'),
    label = cms.untracked.string('BTagMISTAGSSVHPTwp_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHELtable_v5_offline'),
    label = cms.untracked.string('BTagMISTAGTCHELtable_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHELwp_v5_offline'),
    label = cms.untracked.string('BTagMISTAGTCHELwp_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHEMtable_v5_offline'),
    label = cms.untracked.string('BTagMISTAGTCHEMtable_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHEMwp_v5_offline'),
    label = cms.untracked.string('BTagMISTAGTCHEMwp_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('BTagMISTAGTCHPTtable_v5_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPTtable_v5_offline')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('BTagMISTAGTCHPTwp_v5_offline'),
    label = cms.untracked.string('BTagMISTAGTCHPTwp_v5_offline')
    ),
))
PoolDBESSourceMistag110118.connect = 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS'
# foo bar baz
# xQZJFAhVORgUh
